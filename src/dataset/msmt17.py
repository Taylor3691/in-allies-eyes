from base import Object, config
import os.path as osp
from utils import pluck_msmt, print_split_samples


class MSMT17(Object):
    def __init__(self, root: str = config.MSMT_17_DATASET_PATH):
        self._root = root
        self._train, self._val = [], []
        self._query, self._gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        return
    def load(self):
        self._train, train_pids = pluck_msmt(osp.join(self._root, 'list_train.txt'), 'train')
        self._val, val_pids = pluck_msmt(osp.join( self._root, 'list_val.txt'), 'train')
        self._train = self._train + self._val
        self._query, self.query_pids = pluck_msmt(osp.join(self._root, 'list_query.txt'), 'test')
        self._gallery, self.gallery_pids = pluck_msmt(osp.join(self._root, 'list_gallery.txt'), 'test')
        self._num_train_pids = len(list(set(train_pids).union(set(val_pids))))
        return
    
    def save(path):
        return super().save()
    
    def clone():
        return
    
    def info(self):
        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
                .format(self._num_train_pids, len(self._train)))
        print("  query    | {:5d} | {:8d}"
                .format(len(self.query_pids), len(self._query)))
        print("  gallery  | {:5d} | {:8d}"
                .format(len(self.gallery_pids), len(self._gallery)))

        print_split_samples("train", self._train, k=5)
        print_split_samples("query", self._query, k=5)
        print_split_samples("gallery", self._gallery, k=5)
        return
    
    def accept():
        return