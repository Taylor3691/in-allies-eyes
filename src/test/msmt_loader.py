import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from dataset import MSMT17
import cv2

msmt = MSMT17()

msmt.load()

msmt.info()

batch, indices = next(msmt.train_loader)
assert len(batch) > 0, "Empty batch returned"
assert len(batch) == len(indices), "Batch size and indices mismatch"
print(f"First train batch: {len(batch)} images, indices[0]={indices[0]}")
print(f"First image shape: {getattr(batch[0], 'shape', None)}")
cv2.imshow("image", batch[0])
cv2.waitKey(0)