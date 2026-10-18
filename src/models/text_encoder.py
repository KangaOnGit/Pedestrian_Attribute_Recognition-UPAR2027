import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from jaxtyping import Float, Bool
from transformers import Siglip2Tokenizer, AutoModel
from transformers.models.siglip.modeling_siglip import (
    create_bidirectional_mask,
)
from src.modules.self_attn import ClassSelfAttention
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
        model_name: str,
        text_encoder,
        prompts: list[str],
        ctx_len: int,
        csc: bool = True,
        hybrid: bool = False,
        category_ids: list[int] | None = None,
        debug: bool = False,
        freeze_backbone: bool = False
    ):
        super().__init__()

        if not prompts:
            raise ValueError("prompts cannot be empty.")
        if ctx_len < 1:
            raise ValueError("ctx_len must be >= 1.")

        self.tokenizer = Siglip2Tokenizer.from_pretrained(model_name)
        
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
            print(f"Embedding Dim: {self.embed_dim}")
            print(f"Max Length: {self.max_len}")
            print(f"# Special Tokens: {num_special}")
            
        if debug:
            print("Tokenizer BOS:", self.tokenizer.bos_token_id)
            print("Tokenizer EOS:", self.tokenizer.eos_token_id)
            print("Tokenizer PAD:", self.tokenizer.pad_token_id)

            print("Model BOS:", self.text_encoder.config.bos_token_id)
            print("Model EOS:", self.text_encoder.config.eos_token_id)
            print("Model PAD:", self.text_encoder.config.pad_token_id)

            print("Special tokens:", self.tokenizer.special_tokens_map)
            print("EOS token:", repr(self.tokenizer.eos_token))
            print("BOS token:", repr(self.tokenizer.bos_token))
        # for some reason, tokenizer and text encoder has separate token ids
            # son
        self.text_encoder.config.vocab_size = len(self.tokenizer)
        self.text_encoder.config.bos_token_id = self.tokenizer.bos_token_id
        self.text_encoder.config.eos_token_id = self.tokenizer.eos_token_id
        self.text_encoder.config.pad_token_id = self.tokenizer.pad_token_id
        
        if debug:
            print(inspect.signature(self.text_encoder.encoder.forward))
            print(inspect.getsource(self.text_encoder.embeddings.forward))
            print(inspect.getsource(self.text_encoder.encoder.forward))

        self.prompts = prompts
        self.num_classes = len(prompts)
        self.ctx_len = ctx_len
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

            self.category_ctx: Float[torch.Tensor, "6 ctx_len E"] = nn.Parameter(
                torch.empty(
                    self.num_categories,
                    ctx_len,
                    self.embed_dim,
                )
            )
            if debug:
                print(f"category_ctx shape: {self.category_ctx.shape}")

            self.class_ctx: Float[torch.Tensor, "40 ctx_len E"] = nn.Parameter(
                torch.zeros(
                    self.num_classes,
                    ctx_len,
                    self.embed_dim,
                )
            )
            nn.init.normal_(self.category_ctx, std=0.02)
            if debug:
                print(f"class_ctx shape: {self.class_ctx.shape}")

        elif csc:
            # Independent context per class
            self.ctx: Float[torch.Tensor, "40 ctx_len E"] = nn.Parameter(
                torch.empty(
                    self.num_classes,
                    ctx_len,
                    self.embed_dim,
                )
            )
            nn.init.normal_(self.ctx, std=0.02)
            if debug:
                print(f"ctx shape: {self.ctx.shape}")

        else:
            # One context shared by every class.
            self.ctx: Float[torch.Tensor, "1 ctx_len E"] = nn.Parameter(
                torch.empty(1, ctx_len, self.embed_dim)
            )
            nn.init.normal_(self.ctx, std=0.02)
            if debug:
                print(f"ctx shape: {self.ctx.shape}")
            

        self.pad_token_id = self.tokenizer.pad_token_id # 0
        if debug:
            print(f"Pad Token ID: {self.pad_token_id}")
        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer must define a pad_token_id."
            )

        self.eos_token_id = self.tokenizer.eos_token_id # 1
        if debug:
            print(f"EOS Token ID: {self.eos_token_id}")

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

        token_ids: Float[torch.Tensor, "40, stored_len"] = torch.full(
            (self.num_classes, stored_len),
            self.pad_token_id, # Fill with [PAD]
            dtype=torch.long,
        )

        token_mask: Float[torch.Tensor, "40, stored_len"] = torch.zeros(
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
            print(f"Token IDs Shape: {self.token_ids.shape}")
            print(f"Token Masks Shape: {self.token_mask.shape}")
            print(f"Token lengths: {self.token_lengths}")
            
            encoded = self.tokenizer(
                "Age-Young",
                add_special_tokens=True,
            )
            print("IDs:", encoded["input_ids"])
            print(
                "Tokens:",
                self.tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"]
                ),
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
    ) -> Float[torch.Tensor, "num_classes E"]:

        embedding_layer = self.text_encoder.embeddings

        token_ids = self.token_ids
        token_mask = self.token_mask

        # E is embedding dimension
        # Get embedding only
        phrase_embeds: Float[torch.Tensor, "40, stored_len, E"] = embedding_layer.token_embedding(token_ids)
        if self.debug:
            print(f"phrase_embeds shape: {phrase_embeds.shape}")

        sequences = []
        attention_masks = []
        eos_positions = []
        # Basically add ctx vectors to the embedded sentences
        for i in range(self.num_classes):
            length = int(self.token_lengths[i].item())

            ids = token_ids[i, :length]
            embeds = phrase_embeds[i, :length]

            # Find leading special tokens, e.g. BOS if the tokenizer
            # uses one, insert context after those tokens
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

            context = self._get_context(i).to(
                dtype=phrase_embeds.dtype
            )

            # [prefix specials] + [learnable context] + [text and EOS]
            combined = torch.cat(
                [prefix, context, text],
                dim=0,
            )

            combined_mask = torch.ones(
                combined.shape[0],
                device=combined.device,
                dtype=torch.bool,
            )

            combined = combined[:self.max_len]
            combined_mask = combined_mask[:self.max_len]

            pad_length = self.max_len - combined.shape[0]

            if pad_length > 0:
                combined = F.pad(combined, (0, 0, 0, pad_length))
                combined_mask = F.pad(
                    combined_mask,
                    (0, pad_length),
                    value=False,
                )

            # EOS is at the end of the valid sequence when the
            # tokenizer's standard special-token format ends in EOS.
            eos_matches = (ids == self.eos_token_id).nonzero(
                as_tuple=True
            )[0]

            if eos_matches.numel() == 0:
                raise ValueError(f"EOS token missing from prompt {i}")

            eos_pos = int(eos_matches[-1].item()) + self.ctx_len

            if eos_pos >= self.max_len:
                raise ValueError(f"EOS position exceeds sequence length for prompt {i}")

            sequences.append(combined)
            attention_masks.append(combined_mask)
            eos_positions.append(eos_pos)

        inputs_embeds: Float[torch.Tensor, "40 max_len E"] = torch.stack(sequences, dim=0)
        attention_mask: Float[torch.Tensor, "40 max_len"] = torch.stack(attention_masks, dim=0)
        if self.debug:
            print(f"inputs_embeds shape: {inputs_embeds.shape}")
        
        position_ids: Float[torch.Tensor, "40 max_len"] = torch.arange(
            self.max_len,
            device=inputs_embeds.device,).unsqueeze(0).expand(self.num_classes, -1)
        position_embeds: Float[torch.Tensor, "40 max_len E"] = embedding_layer.position_embedding(position_ids)
        if self.debug:
            print(f"position_ids shape: {position_ids.shape}")
            print(f"position_embeds shape: {position_embeds.shape}")
            
        hidden_states: Float[torch.Tensor, "40 max_len E"] = inputs_embeds + position_embeds
        
        # Bidirectional so there's 2 max_len, 1 for forward and 1 for backward
        attention_mask: Float[torch.Tensor, "40 1 max_len max_len"] = create_bidirectional_mask(
            config=self.text_encoder.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        if self.debug:
            print(f"attention_mask bidirectional shape: {attention_mask.shape}")

        eos_positions = torch.tensor(
            eos_positions,
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        
        encoder_outputs = self.text_encoder.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states: Float[torch.Tensor, "40 max_len E"] = self.text_encoder.final_layer_norm(
            encoder_outputs.last_hidden_state
        )
        if self.debug:
            print(f"hidden_states shape: {hidden_states.shape}")
            
        batch_indices: Float[torch.Tensor, "40"] = torch.arange(
            self.num_classes,
            device=hidden_states.device,
        )

        # get [EOS]/last hidden of every prompt
        text_features: Float[torch.Tensor, "40 E"] = hidden_states[batch_indices, eos_positions]
        text_features: Float[torch.Tensor, "40 E"] = self.text_encoder.head(text_features)
        if self.debug:
            print(f"text_features shape: {text_features.shape}")
            
        out: Float[torch.Tensor, "40 E"] = text_features + self.class_attention(text_features)
        if self.debug:
            print(f"out shape: {out.shape}")
        return out


if __name__ == "__main__":
    from src.utils.config import load_config
    P = load_config("configs/prompts.yaml")
    
    te = TextEncoder(
        model_name = "google/siglip2-large-patch16-256",
        text_encoder = AutoModel.from_pretrained("google/siglip2-large-patch16-256").text_model,
        prompts = P["prompts"],
        ctx_len = 16,
        csc = False,
        hybrid = True,
        category_ids = P["category_ids"],
        debug = True
    )
    print(inspect.getsource(te.text_encoder.embeddings.forward))
    print(inspect.getsource(te.text_encoder.forward))
    
    text_features = te.forward()