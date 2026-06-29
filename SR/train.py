import os
import sys
from pathlib import Path


def main() -> None:
    restormer_root = Path(__file__).resolve().parents[1]
    os.chdir(restormer_root)
    if str(restormer_root) not in sys.path:
        sys.path.insert(0, str(restormer_root))

    from basicsr.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
