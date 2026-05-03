import os.path as osp
from itertools import islice


def first_sample_basenames(data, k: int = 5):
	"""Lấy tối đa k tên file (basename) từ một split của dataset.

	Mô tả:
		Hàm này duyệt các phần tử đầu tiên của `data` và trích ra tên file ảnh
		(không bao gồm đường dẫn) từ phần tử thứ 0 của mỗi item.

	Input:
		- data: iterable (thường là list) các item, mỗi item là tuple/list có phần
		  tử đầu tiên là đường dẫn (string/Path).
		- k (int): số lượng sample muốn lấy.

	Output:
		- List[str]: danh sách tối đa k tên file (ví dụ: "0001_c1s1_....jpg").
	"""
	names = []
	for item in islice(data, k):
		if not item:
			continue
		names.append(osp.basename(str(item[0])))
	return names


def print_split_samples(split_name: str, data, k: int = 5):
	"""In ra tối đa k sample tên ảnh (chỉ basename) cho một split.

	Mô tả:
		In theo format dễ nhìn: 1 dòng tiêu đề cho split, sau đó mỗi sample 1 dòng.

	Input:
		- split_name (str): tên split (vd: "train", "query", "gallery").
		- data: iterable các item (tuple/list) với phần tử đầu tiên là đường dẫn ảnh.
		- k (int): số lượng sample muốn in.

	Output:
		- None (hàm chỉ in ra console).
	"""
	names = first_sample_basenames(data, k=k)
	if not names:
		print(f"  {split_name} samples (first 0): []")
		return

	print(f"  {split_name} samples (first {len(names)}):")
	for i, name in enumerate(names, start=1):
		print(f"    {i}. {name}")

