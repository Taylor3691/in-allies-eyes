import re
from pathlib import Path
from base import Object
import os.path as osp
import glob

def process_dir(dataset: Object, dir_path: Path, relabel: bool):
    items = []
    pid_set = set()
    pattern = re.compile(r"([-\d]+)_c(\d)")

    for img_path in dir_path.glob("*.jpg"):
        m = pattern.search(img_path.name)
        if not m:
            continue
        pid = int(m.group(1))
        camid = int(m.group(2)) - 1
        if pid == -1:
            continue
        pid_set.add(pid)

    print(len(pid_set))

    pid2label = {pid: i for i, pid in enumerate(sorted(pid_set))}
    
    for img_path in dir_path.glob("*.jpg"):
        m = pattern.search(img_path.name)
        if not m:
            continue
        pid, camid = map(int, m.groups())
        if pid == -1:
            continue
        assert 0 <= pid <= 1501
        assert 1 <= camid <= 6
        camid -= 1

        if relabel:
            pid = pid2label[pid]
        
        items.append((str(img_path), pid, camid))

    return items

def check_before_run(dataset: Object):
    if not osp.exists(dataset.dataset_dir):
        raise RuntimeError("'{}' is not available".format(dataset.dataset_dir))
    if not osp.exists(dataset.train_dir):
        raise RuntimeError("'{}' is not available".format(dataset.train_dir))
    if not osp.exists(dataset.query_dir):
        raise RuntimeError("'{}' is not available".format(dataset.query_dir))
    if not osp.exists(dataset.gallery_dir):
        raise RuntimeError("'{}' is not available".format(dataset.gallery_dir))
    
    return