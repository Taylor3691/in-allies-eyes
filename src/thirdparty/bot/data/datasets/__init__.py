# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
# from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
# from .veri import VeRi
from .grid import GRID

__factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    # 'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    # 'veri': VeRi,
    'grid': GRID
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if isinstance(name, (list, tuple)):
        name = name[0]
    if name.startswith('grid'):
        if name == 'grid_custom':
            split_id = 'custom'
        elif '_' in name:
            split_id = int(name.split('_')[1])
        else:
            split_id = 0
        return GRID(split_id=split_id, *args, **kwargs)

    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
