import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from dataset import Market1501


market1501 = Market1501()

market1501.load()

print(market1501.train_dir)
print(market1501.gallery_dir)
print(market1501.query_dir)


print(len(market1501._train))
print(len(market1501._gallery))
print(len(market1501._query))