import shutil
import zipfile
from pathlib import Path

wheels = Path("wheels")
target = Path("vendor_linux")
target.mkdir(exist_ok=True)

required = {
    "transformers": "transformers-4.56.2-py3-none-any.whl",
    "huggingface_hub": "huggingface_hub-0.36.2-py3-none-any.whl",
    "tokenizers": (
        "tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl"
    ),
}

# Verify all required wheels exist before deleting anything.
for package, filename in required.items():
    if not (wheels / filename).is_file():
        raise FileNotFoundError(f"Missing wheel: {wheels / filename}")

# Remove only the three packages being replaced and their metadata.
for package in required:
    for path in target.iterdir():
        if path.name == package or (
            path.name.startswith(package + "-")
            and path.name.endswith(".dist-info")
        ):
            if path.is_dir():
                shutil.rmtree(path)
                print("REMOVED:", path)
            else:
                path.unlink()
                print("REMOVED:", path)

# Extract the exact versions, not the first wildcard match.
for package, filename in required.items():
    wheel_path = wheels / filename
    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(target)
    print("EXTRACTED:", wheel_path)

print("\nReplacement complete.")
