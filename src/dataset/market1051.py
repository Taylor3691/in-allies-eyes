from base import Object, config
from pathlib import Path
from utils import process_dir, get_metadata

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

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = get_metadata(self._train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = get_metadata(self._query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = get_metadata(self._gallery)

        return 
    def save(self):
        return super().save()
    
    def clone():
        pass

    def info(self):
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams))
        print("  ----------------------------------------")
        return
    
    def accept():
        return