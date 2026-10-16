import shutil
import zipfile
from pathlib import Path

wheels = Path("wheels")
target = Path("vendor_linux")
wheel = wheels / (
    "opencv_python_headless-4.10.0.84-"
    "cp37-abi3-manylinux_2_17_x86_64."
    "manylinux2014_x86_64.whl"
)

if not wheel.is_file():
    matches = list(wheels.glob(
        "opencv_python_headless-4.10.0.84-*.whl"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one 4.10.0.84 wheel; found: {matches}"
        )
    wheel = matches[0]

for name in ("cv2",):
    path = target / name
    if path.exists():
        shutil.rmtree(path)

for path in list(target.iterdir()):
    if path.is_dir() and (
        path.name.startswith("opencv_python.libs")
        or path.name.startswith("opencv_python_headless.libs")
        or path.name.startswith("opencv_python-")
        or path.name.startswith("opencv_python_headless-")
    ):
        shutil.rmtree(path)
    elif path.is_file() and path.name.startswith("opencv_python"):
        path.unlink()

with zipfile.ZipFile(wheel) as archive:
    archive.extractall(target)

print("Extracted:", wheel)
print("cv2 exists:", (target / "cv2").is_dir())
print("OpenCV metadata:")
for path in sorted(target.glob("opencv_python*.dist-info")):
    print(" ", path.name)
