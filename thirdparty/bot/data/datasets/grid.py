# encoding: utf-8
import os.path as osp
import random
import re

from .bases import BaseImageDataset


class GRID(BaseImageDataset):
    dataset_dir = 'grid'

    def __init__(self, root='/home/haoluo/data', split_id=0, verbose=True, **kwargs):
        super(GRID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir, 'underground_reid')
        self.split_id = split_id

        # Path configurations
        self.txt_path = osp.join(self.dataset_dir, 'correspondence.txt')

        self._check_before_run()

        # Read correspondence.txt to map identities to filenames
        with open(self.txt_path, 'r') as f:
            lines = f.read().splitlines()

        pattern = re.compile(r'(\d+)_(\d)_')

        # ---- Build image lists from correspondence.txt ----
        # Probe images: 250 matched identities (pids 1-250), one image each
        probe_images = []  # list of (path, pid, camid)
        for k in range(1, 251):
            line = lines[k + 254].strip()
            filename = line.split(" ")[0]
            pid, camid = map(int, pattern.search(filename).groups())
            probe_images.append((
                osp.join(self.dataset_dir, 'probe', filename),
                pid, camid - 1
            ))

        # Gallery images: 250 matched identities (pids 1-250), one image each
        gallery_matched = []  # list of (path, pid, camid)
        for k in range(1, 251):
            line = lines[k + 1].strip()
            filename = line.split(" ")[0]
            pid, camid = map(int, pattern.search(filename).groups())
            gallery_matched.append((
                osp.join(self.dataset_dir, 'gallery', filename),
                pid, camid - 1
            ))

        # Distractor images: 775 images with pid=0
        distractors = []  # list of (path, pid=-1, camid)
        for d in range(1, 776):
            line = lines[d + 507].strip()
            filename = line.split(" ")[0]
            _, camid = map(int, pattern.search(filename).groups())
            distractors.append((
                osp.join(self.dataset_dir, 'gallery', filename),
                -1, camid - 1
            ))

        # ---- Choose splitting strategy ----
        if split_id == 'custom':
            train, query, gallery = self._custom_split(
                probe_images, gallery_matched, distractors
            )
        else:
            train, query, gallery = self._fold_split(
                split_id, probe_images, gallery_matched, distractors, lines, pattern
            )

        if verbose:
            print("=> GRID (split {}) loaded".format(self.split_id))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _custom_split(self, probe_images, gallery_matched, distractors,
                      train_ratio=0.8, seed=42):
        """Create an 80/20 train/test split over the 250 paired identities."""
        all_pids = list(range(250))  # indices into probe_images / gallery_matched
        rng = random.Random(seed)
        rng.shuffle(all_pids)

        num_train = int(len(all_pids) * train_ratio)
        train_indices = sorted(all_pids[:num_train])
        test_indices = sorted(all_pids[num_train:])

        # Relabel training PIDs to contiguous 0..N-1
        pid2label = {}
        for new_label, idx in enumerate(train_indices):
            pid2label[probe_images[idx][1]] = new_label

        # Training set: both probe and gallery images for train identities
        train = []
        for idx in train_indices:
            p_path, p_pid, p_cam = probe_images[idx]
            g_path, g_pid, g_cam = gallery_matched[idx]
            train.append((p_path, pid2label[p_pid], p_cam))
            train.append((g_path, pid2label[g_pid], g_cam))

        # Query set: probe images for test identities (original PIDs)
        query = []
        for idx in test_indices:
            p_path, p_pid, p_cam = probe_images[idx]
            query.append((p_path, p_pid, p_cam))

        # Gallery set: gallery images for test identities + all distractors
        gallery = []
        for idx in test_indices:
            g_path, g_pid, g_cam = gallery_matched[idx]
            gallery.append((g_path, g_pid, g_cam))
        gallery.extend(distractors)

        return train, query, gallery

    def _fold_split(self, split_id, probe_images, gallery_matched, distractors,
                    lines, pattern):
        """Use the original 10-fold cross-validation from the .mat file."""
        import scipy.io as sio

        mat_path = osp.join(self.dataset_dir, 'features_and_partitions.mat')
        mat_data = sio.loadmat(mat_path)
        test_idx_all = mat_data['testIdxAll']
        train_idx_all = mat_data['trainIdxAll']

        fold_test = test_idx_all[0, split_id]
        fold_train = train_idx_all[0, split_id]

        # Re-build the MATLAB index -> (path, filename) mapping needed for fold splits
        with open(self.txt_path, 'r') as f:
            all_lines = f.read().splitlines()

        idx_to_name = {}
        for k in range(1, 251):
            idx = 2 * k - 1
            line = all_lines[k + 1].strip()
            filename = line.split(" ")[0]
            idx_to_name[idx] = (osp.join(self.dataset_dir, 'gallery', filename), filename)

        for k in range(1, 251):
            idx = 2 * k
            line = all_lines[k + 254].strip()
            filename = line.split(" ")[0]
            idx_to_name[idx] = (osp.join(self.dataset_dir, 'probe', filename), filename)

        for d in range(1, 776):
            idx = 500 + d
            line = all_lines[d + 507].strip()
            filename = line.split(" ")[0]
            idx_to_name[idx] = (osp.join(self.dataset_dir, 'gallery', filename), filename)

        # Training set
        train_probe_indices = fold_train['probeImIdx'][0, 0].flatten()
        train_gallery_indices = fold_train['galleryImIdx'][0, 0].flatten()
        train_pids = fold_train['personIdx'][0, 0].flatten()

        pid2label = {pid: label for label, pid in enumerate(sorted(list(set(train_pids))))}

        train = []
        for idx in train_probe_indices:
            img_path, filename = idx_to_name[idx]
            pid, camid = map(int, pattern.search(filename).groups())
            pid = pid2label[pid]
            camid -= 1
            train.append((img_path, pid, camid))

        for idx in train_gallery_indices:
            img_path, filename = idx_to_name[idx]
            pid, camid = map(int, pattern.search(filename).groups())
            pid = pid2label[pid]
            camid -= 1
            train.append((img_path, pid, camid))

        # Query set
        test_probe_indices = fold_test['probeImIdx'][0, 0].flatten()
        query = []
        for idx in test_probe_indices:
            img_path, filename = idx_to_name[idx]
            pid, camid = map(int, pattern.search(filename).groups())
            camid -= 1
            query.append((img_path, pid, camid))

        # Gallery set
        test_gallery_indices = fold_test['galleryImIdx'][0, 0].flatten()
        gallery = []
        for idx in test_gallery_indices:
            img_path, filename = idx_to_name[idx]
            pid, camid = map(int, pattern.search(filename).groups())
            if pid == 0:
                pid = -1
            camid -= 1
            gallery.append((img_path, pid, camid))

        return train, query, gallery

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.txt_path):
            raise RuntimeError("'{}' is not available".format(self.txt_path))
