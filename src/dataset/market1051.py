from base import Object, config
from pathlib import Path
from utils import process_dir

class Market1501(Object):

    def __init__(self, path: str = config.MARKET_1501_DATASET_PATH):
        self.root = Path(config.MARKET_1501_DATASET_PATH)

        self.train_dir = self.root / "bounding_box_train"
        self.query_dir = self.root / "query"
        self.gallery_dir = self.root / "bounding_box_test"

        self._train = []
        self._gallery = []
        self._query = []

        return
    def load(self):
        self._train = process_dir(self.train_dir, relabel=True)
        self._query = process_dir(self.query_dir, relabel= False)
        self._gallery = process_dir(self.gallery_dir, relabel= False)

        return 
    def save(self):
        return super().save()
    
    def clone():
        pass

    def info():
        return
    
    def accept():
        return