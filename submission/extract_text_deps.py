import zipfile
from pathlib import Path

wheels = Path("wheels")
target = Path("vendor_linux")

packages = ("ftfy", "wcwidth")

for package in packages:
    matches = sorted(wheels.glob(f"{package}-*.whl"))
    if not matches:
        raise FileNotFoundError(f"No wheel found for {package}")

    wheel = matches[-1]
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)

    print("EXTRACTED:", wheel)
