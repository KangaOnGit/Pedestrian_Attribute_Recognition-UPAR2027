from src.train.augmentation import build_transforms
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(name)s | %(funcName)s: %(message)s"
)

if __name__ == "__main__":
    aug = build_transforms()
