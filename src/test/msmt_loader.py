import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from dataset import MSMT17

msmt = MSMT17()

msmt.load()