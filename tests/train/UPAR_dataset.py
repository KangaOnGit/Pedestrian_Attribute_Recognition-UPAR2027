from src.train.load_data import UPAR_dataset
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(name)s | %(funcName)s: %(message)s"
)

if __name__ == "__main__":
    data = UPAR_dataset("data/annotations/train.csv")
