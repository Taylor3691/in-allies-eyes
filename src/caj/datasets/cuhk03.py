from __future__ import print_function, absolute_import
import os.path as osp
import json

from ..utils.data import BaseImageDataset


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


class CUHK03(BaseImageDataset):
    """
    CUHK03
    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    """
    dataset_dir = 'cuhk03'

    def __init__(self, root, split_id=0, cuhk03_labeled=False, cuhk03_classic_split=False, verbose=True, **kwargs):
        super(CUHK03, self).__init__()
        self.dataset_dir = root
        self.data_dir = osp.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = osp.join(self.dataset_dir, 'splits_classic_detected.json')
        self.split_classic_lab_json_path = osp.join(self.dataset_dir, 'splits_classic_labeled.json')

        self.split_new_det_json_path = osp.join(self.dataset_dir, 'splits_new_detected.json')
        self.split_new_lab_json_path = osp.join(self.dataset_dir, 'splits_new_labeled.json')

        self.split_new_det_mat_path = osp.join(self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat')
        self.split_new_lab_mat_path = osp.join(self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat')

        self._check_before_run()

        if cuhk03_labeled:
            image_type = 'labeled'
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            image_type = 'detected'
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(splits), "Condition split_id ({}) < len(splits) ({}) is false".format(split_id,
                                                                                                    len(splits))
        split = splits[split_id]

        def _correct_paths(dataset_list, target_dir):
            corrected = []
            for item in dataset_list:
                img_path, pid, camid = item
                img_path = img_path.replace('\\', '/')
                img_name = osp.basename(img_path)
                corrected.append((osp.join(target_dir, img_name), int(pid), int(camid)))
            return corrected

        imgs_dir = self.imgs_labeled_dir if cuhk03_labeled else self.imgs_detected_dir
        train = _correct_paths(split['train'], imgs_dir)
        query = _correct_paths(split['query'], imgs_dir)
        gallery = _correct_paths(split['gallery'], imgs_dir)

        if verbose:
            print("=> CUHK03 ({}) loaded".format(image_type))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.raw_mat_path):
            raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
        if not osp.exists(self.split_new_det_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_det_mat_path))
        if not osp.exists(self.split_new_lab_mat_path):
            raise RuntimeError("'{}' is not available".format(self.split_new_lab_mat_path))
