import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train import train_and_select_best_model


if __name__ == "__main__":
    best_name, _, results = train_and_select_best_model()
    print(f"Best model: {best_name}")
    print(f"Validation F1: {results[best_name]['validation']['f1']:.4f}")
    print(f"Test F1: {results[best_name]['test']['f1']:.4f}")
