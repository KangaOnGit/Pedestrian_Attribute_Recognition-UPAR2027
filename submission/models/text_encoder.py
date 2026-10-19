import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from transformers.masking_utils import create_bidirectional_mask
from modules.self_attn import ClassSelfAttention
from modules.visual_residual import VisualResidualGate
from models.pretrained import PRETRAINED_MODEL_PATH

log = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    """
    SigLIP 2 text encoder with configurable CoOp context.

    Modes:
        csc=True, hybrid=False:
            Class-specific context.
            ctx shape: [num_classes, ctx_len, embed_dim]

        csc=False, hybrid=False:
            One shared context across all classes.
            ctx shape: [1, ctx_len, embed_dim]

        hybrid=True:
            Category context + class-specific residual.
            category_ctx: [num_categories, ctx_len, embed_dim]
            class_ctx:    [num_classes, ctx_len, embed_dim]

    Output:
        [num_classes, embed_dim]
        These are pooled text hidden states, not necessarily the
        final projected SigLIP contrastive embeddings
    """

    def __init__(
        self,
        model_path: str | Path,
        text_encoder,
        prompts: list[str],
        ctx_len: int,
        csc: bool = True,
        hybrid: bool = False,
        category_ids: list[int] | None = None,
        debug: bool = False,
        freeze_backbone: bool = False,
        condition_text: bool = False
    ):
        super().__init__()

        if not prompts:
            raise ValueError("prompts cannot be empty.")
        if ctx_len < 1:
            raise ValueError("ctx_len must be >= 1.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        
        self.text_encoder = text_encoder
        self.embed_dim = self.text_encoder.get_input_embeddings().embedding_dim
        self.max_len = self.text_encoder.config.max_position_embeddings
        num_special = self.tokenizer.num_special_tokens_to_add(pair=False)
        
        self.class_attention = ClassSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=16,
            dropout=0.1,
        )
        
        if debug:
            log.debug("Embedding Dim: %s", self.embed_dim)
            log.debug("Max Length: %s", self.max_len)
            log.debug("# Special Tokens: %s", num_special)
            
        if debug:
            log.debug("Tokenizer BOS: %s", self.tokenizer.bos_token_id)
            log.debug("Tokenizer EOS: %s", self.tokenizer.eos_token_id)
            log.debug("Tokenizer PAD: %s", self.tokenizer.pad_token_id)

            log.debug("Model BOS: %s", self.text_encoder.config.bos_token_id)
            log.debug("Model EOS: %s", self.text_encoder.config.eos_token_id)
            log.debug("Model PAD: %s", self.text_encoder.config.pad_token_id)

            log.debug("Special tokens: %s", self.tokenizer.special_tokens_map)
            log.debug("EOS token: %r", self.tokenizer.eos_token)
            log.debug("BOS token: %r", self.tokenizer.bos_token)
        # for some reason, tokenizer and text encoder has separate token ids
            # son
        self.text_encoder.config.vocab_size = len(self.tokenizer)
        self.text_encoder.config.bos_token_id = self.tokenizer.bos_token_id
        self.text_encoder.config.eos_token_id = self.tokenizer.eos_token_id
        self.text_encoder.config.pad_token_id = self.tokenizer.pad_token_id
        
        if debug:
            log.debug("%s", inspect.signature(self.text_encoder.encoder.forward))
            log.debug("%s", inspect.getsource(self.text_encoder.embeddings.forward))
            log.debug("%s", inspect.getsource(self.text_encoder.encoder.forward))

        self.prompts = prompts
        self.num_classes = len(prompts)
        
        self.ctx_len = ctx_len
        self.condition_text = condition_text
        if condition_text:
            self.visual_residual = VisualResidualGate(
                shallow_dim=self.embed_dim,
                ctx_len=self.ctx_len,
                embed_dim=self.text_encoder.config.hidden_size,
                debug=debug,
            )
            
        self.csc = csc
        self.hybrid = hybrid
        
        self.debug = debug

        if hybrid:
            if category_ids is None:
                raise ValueError(
                    "category_ids is required when hybrid=True."
                )

            if len(category_ids) != self.num_classes:
                raise ValueError(
                    "category_ids must contain one category ID "
                    "per prompt."
                )

            if min(category_ids) < 0:
                raise ValueError("Category IDs must be non-negative.")

            unique_ids = sorted(set(category_ids))
            id_map = {
                old_id: new_id
                for new_id, old_id in enumerate(unique_ids)
            }
            mapped_ids = [id_map[x] for x in category_ids]

            self.num_categories = len(unique_ids)

            self.register_buffer(
                "category_ids",
                torch.tensor(mapped_ids, dtype=torch.long),
            )

            self.category_ctx: torch.Tensor = nn.Parameter(
                torch.empty(
                    self.num_categories,
                    ctx_len,
                    self.embed_dim,
                )
            )
            if debug:
                log.debug("category_ctx shape: %s", self.category_ctx.shape)

            self.class_ctx: torch.Tensor = nn.Parameter(
                torch.zeros(
                    self.num_classes,
                    ctx_len,
                    self.embed_dim,
                )
            )
            nn.init.normal_(self.category_ctx, std=0.02)
            if debug:
                log.debug("class_ctx shape: %s", self.class_ctx.shape)

        elif csc:
            # Independent context per class
            self.ctx: torch.Tensor = nn.Parameter(
                torch.empty(
                    self.num_classes,
                    ctx_len,
                    self.embed_dim,
                )
            )
            nn.init.normal_(self.ctx, std=0.02)
            if debug:
                log.debug("ctx shape: %s", self.ctx.shape)

        else:
            # One context shared by every class.
            self.ctx: torch.Tensor = nn.Parameter(
                torch.empty(1, ctx_len, self.embed_dim)
            )
            nn.init.normal_(self.ctx, std=0.02)
            if debug:
                log.debug("ctx shape: %s", self.ctx.shape)
            

        self.pad_token_id = self.tokenizer.pad_token_id # 0
        if debug:
            log.debug("Pad Token ID: %s", self.pad_token_id)
        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer must define a pad_token_id."
            )

        self.eos_token_id = self.tokenizer.eos_token_id # 1
        if debug:
            log.debug("EOS Token ID: %s", self.eos_token_id)

        # Tokenize prompts with special tokens.
        tokenized = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        raw_tokens = tokenized["input_ids"]
        stored_len = self.max_len - ctx_len
        if stored_len < 2:
            raise ValueError(
                "Not enough space for prompt and special tokens"
            )

        token_ids: torch.Tensor = torch.full(
            (self.num_classes, stored_len),
            self.pad_token_id, # Fill with [PAD]
            dtype=torch.long,
        )

        token_mask: torch.Tensor = torch.zeros(
            (self.num_classes, stored_len),
            dtype=torch.long,
        )

        lengths = []
        for i, ids in enumerate(raw_tokens):
            if self.eos_token_id not in ids:
                raise ValueError(f"Prompt {i} has no EOS token.")

            # Preserve EOS while truncating
            if len(ids) > stored_len:
                ids = ids[:stored_len]
                ids[-1] = self.eos_token_id

            length = len(ids)
            lengths.append(length)

            token_ids[i, :length] = torch.tensor(
                ids,
                dtype=torch.long,
            )

            token_mask[i, :length] = 1

        self.register_buffer("token_ids", token_ids)
        self.register_buffer("token_mask", token_mask)
        self.register_buffer(
            "token_lengths",
            torch.tensor(lengths, dtype=torch.long),
        )
        if debug:
            log.debug("Token IDs Shape: %s", self.token_ids.shape)
            log.debug("Token Masks Shape: %s", self.token_mask.shape)
            log.debug("Token lengths: %s", self.token_lengths)
            
            encoded = self.tokenizer(
                "Age-Young",
                add_special_tokens=True,
            )
            log.debug("IDs: %s", encoded["input_ids"])
            log.debug(
                "Tokens: %s",
                self.tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
            )
            
        if freeze_backbone:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
    
    def _get_context(self, i: int) -> torch.Tensor:
        """Return the context for attribute i: [ctx_len, embed_dim]."""

        if self.hybrid:
            category_context = self.category_ctx[
                self.category_ids[i]
            ]
            class_residual = self.class_ctx[i]

            return category_context + class_residual

        if self.csc:
            return self.ctx[i]

        return self.ctx[0]


    def forward(
        self,
        shallow_features: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.condition_text and shallow_features is None:
            raise ValueError(
                "shallow_features is required when condition_text=True."
            )

        embedding_layer = self.text_encoder.embeddings

        token_ids = self.token_ids
        token_mask = self.token_mask

        phrase_embeds: torch.Tensor = (
            embedding_layer.token_embedding(token_ids)
        )
        if self.debug:
            log.debug("phrase_embeds shape: %s", phrase_embeds.shape)

        if self.condition_text:
            B = shallow_features.shape[0]
            visual_residuals: torch.Tensor = (
                self.visual_residual(shallow_features).to(
                    dtype=phrase_embeds.dtype,
                    device=phrase_embeds.device,
                )
            )
        else:
            B = 1

        sequences = []
        attention_masks = []
        eos_positions = []

        for i in range(self.num_classes):
            length = int(self.token_lengths[i].item())

            ids = token_ids[i, :length]
            embeds = phrase_embeds[i, :length]

            # Find leading special tokens, e.g. BOS if present.
            special_mask = self.tokenizer.get_special_tokens_mask(
                ids.tolist(),
                already_has_special_tokens=True,
            )

            prefix_len = 0
            for is_special in special_mask:
                if is_special:
                    prefix_len += 1
                else:
                    break

            prefix = embeds[:prefix_len]
            text = embeds[prefix_len:]

            context = self._get_context(i).to(dtype=phrase_embeds.dtype)

            if self.condition_text:
                # [ctx_len, E] + [B, ctx_len, E] -> [B, ctx_len, E]
                context = context.unsqueeze(0) + visual_residuals

                # [B, prefix_len, E] + [B, ctx_len, E] + [B, text_len, E]
                combined = torch.cat(
                    [
                        prefix.unsqueeze(0).expand(B, -1, -1),
                        context,
                        text.unsqueeze(0).expand(B, -1, -1),
                    ],
                    dim=1,
                )
            else:
                # [prefix_len + ctx_len + text_len, E]
                combined = torch.cat(
                    [prefix, context, text],
                    dim=0,
                )

            combined_mask: torch.Tensor = torch.ones(
                combined.shape[:-1],
                device=combined.device,
                dtype=torch.bool,
            )

            seq_dim = 1 if self.condition_text else 0

            combined = combined.narrow(
                seq_dim, 0, min(combined.shape[seq_dim], self.max_len)
            )
            combined_mask = combined_mask.narrow(
                seq_dim, 0, min(combined_mask.shape[seq_dim], self.max_len)
            )

            pad_length = self.max_len - combined.shape[seq_dim]

            if pad_length > 0:
                combined = F.pad(combined, (0, 0, 0, pad_length))
                combined_mask = F.pad(
                    combined_mask,
                    (0, pad_length),
                    value=False,
                )

            eos_matches = (ids == self.eos_token_id).nonzero(as_tuple=True)[0]
            if eos_matches.numel() == 0:
                raise ValueError(f"EOS token missing from prompt {i}")

            eos_pos = int(eos_matches[-1].item()) + self.ctx_len

            if eos_pos >= self.max_len:
                raise ValueError(
                    f"EOS position exceeds sequence length for prompt {i}"
                )

            sequences.append(combined)
            attention_masks.append(combined_mask)
            eos_positions.append(eos_pos)

        if self.condition_text:
            # [B, K, max_len, E]
            inputs_embeds: torch.Tensor = (
                torch.stack(sequences, dim=1)
            )
            attention_mask: torch.Tensor = (
                torch.stack(attention_masks, dim=1)
            )

            # Flatten image and class dimensions for the text transformer.
            inputs_embeds = inputs_embeds.reshape(
                B * self.num_classes,
                self.max_len,
                phrase_embeds.shape[-1],
            )
            attention_mask = attention_mask.reshape(
                B * self.num_classes,
                self.max_len,
            )
        else:
            # [K, max_len, E]
            inputs_embeds: torch.Tensor = (
                torch.stack(sequences, dim=0)
            )
            attention_mask: torch.Tensor = (
                torch.stack(attention_masks, dim=0)
            )

        if self.debug:
            log.debug("inputs_embeds shape: %s", inputs_embeds.shape)

        position_ids: torch.Tensor = torch.arange(
            self.max_len,
            device=inputs_embeds.device,
        ).unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        position_embeds: torch.Tensor = (
            embedding_layer.position_embedding(position_ids)
        )

        if self.debug:
            log.debug("position_ids shape: %s", position_ids.shape)
            log.debug("position_embeds shape: %s", position_embeds.shape)

        hidden_states = inputs_embeds + position_embeds

        attention_mask = create_bidirectional_mask(
            config=self.text_encoder.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        if self.debug:
            log.debug("attention_mask bidirectional shape: %s", attention_mask.shape)

        eos_positions = torch.tensor(
            eos_positions,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        encoder_outputs = self.text_encoder.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states: torch.Tensor = (
            self.text_encoder.final_layer_norm(
                encoder_outputs.last_hidden_state
            )
        )

        if self.debug:
            log.debug("hidden_states shape: %s", hidden_states.shape)

        if self.condition_text:
            hidden_states = hidden_states.reshape(
                B,
                self.num_classes,
                self.max_len,
                hidden_states.shape[-1],
            )

            batch_indices: torch.Tensor = torch.arange(
                B, device=hidden_states.device
            )[:, None]
            class_indices: torch.Tensor = torch.arange(
                self.num_classes, device=hidden_states.device
            )[None, :]

            text_features: torch.Tensor = hidden_states[
                batch_indices,
                class_indices,
                eos_positions[None, :],
            ]
        else:
            class_indices: torch.Tensor = torch.arange(
                self.num_classes,
                device=hidden_states.device,
            )

            text_features: torch.Tensor = hidden_states[
                class_indices, eos_positions
            ]

        text_features = text_features + self.class_attention(text_features)
        if self.debug:
            log.debug("text_features shape: %s", text_features.shape)

        out = self.text_encoder.head(text_features)

        if self.debug:
            log.debug("out shape: %s", out.shape)

        return out

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from src.utils.config import load_config
    P = load_config("configs/prompts.yaml")
    
    te = TextEncoder(
        model_path = PRETRAINED_MODEL_PATH,
        text_encoder = AutoModel.from_pretrained(
            str(PRETRAINED_MODEL_PATH),
            local_files_only=True,
        ).text_model,
        prompts = P["prompts"],
        ctx_len = 16,
        csc = False,
        hybrid = True,
        category_ids = P["category_ids"],
        debug = True,
        condition_text = False,
    )
    log.debug("%s", inspect.getsource(te.text_encoder.embeddings.forward))
    log.debug("%s", inspect.getsource(te.text_encoder.forward))
    
    text_features = te.forward()