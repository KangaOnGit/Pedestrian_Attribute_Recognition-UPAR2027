from pathlib import Path
import yaml

from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(path):
    with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    from huggingface_hub import scan_cache_dir
    print(scan_cache_dir())

def resolve_path(path: Path) -> Path:
    """Resolve relative paths relative to the project root."""
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path