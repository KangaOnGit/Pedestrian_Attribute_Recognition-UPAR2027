import glob
import zipfile
from pathlib import Path

wheels = Path("wheels")
target = Path("vendor_linux")
target.mkdir(exist_ok=True)

patterns = [
    "pyyaml-*.whl",
    "tokenizers-*.whl",
    "safetensors-*.whl",
    "regex-*.whl",
    "huggingface_hub-*.whl",
    "requests-*.whl",
    "tqdm-*.whl",
    "packaging-*.whl",
    "filelock-*.whl",
    "numpy-*.whl",
    "pillow-*.whl",
    "opencv_python-*.whl",
]

for pattern in patterns:
    matches = glob.glob(str(wheels / pattern))
    if not matches:
        print("MISSING:", pattern)
        continue

    # Do not bundle evaluator-provided runtime packages.
    if pattern.startswith(("numpy-", "pillow-")):
        print("SKIP evaluator-provided package:", matches[0])
        continue

    path = matches[0]
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    print("EXTRACTED:", path)
