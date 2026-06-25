# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
# from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
# from .veri import VeRi
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    # 'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    # 'veri': VeRi,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if isinstance(name, (list, tuple)):
        name = name[0]
    if name.startswith('cuhk03'):
        cuhk03_labeled = 'labeled' in name
        return CUHK03(cuhk03_labeled=cuhk03_labeled, *args, **kwargs)

    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
