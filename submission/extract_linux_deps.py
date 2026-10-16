import glob
import zipfile
from pathlib import Path

target = Path("vendor_linux")
wheels = Path("wheels")

patterns = [
    "huggingface_hub-*.whl",
    "requests-*.whl",
    "tqdm-*.whl",
    "packaging-*.whl",
    "filelock-*.whl",
    "fsspec-*.whl",
    "typing_extensions-*.whl",
    "jinja2-*.whl",
    "markupsafe-*.whl",
    "pyyaml-*.whl",
    "certifi-*.whl",
    "charset_normalizer-*.whl",
    "idna-*.whl",
    "urllib3-*.whl",
    "hf_xet-*.whl",
    "tokenizers-*.whl",
    "safetensors-*.whl",
    "regex-*.whl",
    "numpy-*.whl",
    "pillow-*.whl",
]

for pattern in patterns:
    matches = glob.glob(str(wheels / pattern))
    if not matches:
        print("MISSING:", pattern)
        continue
    if pattern.startswith(("numpy-", "pillow-")):
        continue
    with zipfile.ZipFile(matches[0]) as archive:
        archive.extractall(target)
    print("EXTRACTED:", matches[0])
