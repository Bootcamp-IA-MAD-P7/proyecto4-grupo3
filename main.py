import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logging.getLogger("optuna").setLevel(logging.WARNING)

from src.pipeline import main

if __name__ == "__main__":
    main()
