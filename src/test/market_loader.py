import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import cv2
from dataset import Market1501


market1501 = Market1501()

market1501.load()

market1501.info()

batch, indices = next(market1501.train_loader())
assert len(batch) > 0, "Empty batch returned"
assert len(batch) == len(indices), "Batch size and indices mismatch"
print(f"First train batch: {len(batch)} images, indices[0]={indices[0]}")
print(f"First image shape: {getattr(batch[0], 'shape', None)}")
cv2.imshow("image", batch[0])
cv2.waitKey(0)