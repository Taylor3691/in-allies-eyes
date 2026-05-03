from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_FOLDER_RAW = str(_PROJECT_ROOT / "data")

MARKET_1501_DATASET_PATH = str(Path(PATH_FOLDER_RAW) / "market1501" / "Market-1501-v15.09.15")
MSMT_17_DATASET_PATH = str(Path(PATH_FOLDER_RAW) / "msmt17" / "MSMT17_V1")
BATCH_SIZE = 64
