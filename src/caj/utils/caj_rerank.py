import csv
import gc
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def _topk_rank_from_features(features, k, cam_labels=None, same_camera=None, block_size=512):
    N = features.shape[0]
    rank = np.empty((N, k), dtype=np.int32)
    features_t = features.T
    labels = None if cam_labels is None else np.asarray(cam_labels)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        dist = 2 - 2 * np.matmul(features[start:end], features_t)
        if labels is not None:
            cam_mask = labels[start:end, None] == labels[None, :]
            if same_camera:
                dist[~cam_mask] += 999.0
            else:
                dist[cam_mask] += 999.0
        kth = np.arange(k)
        rank[start:end] = np.argpartition(dist, kth, axis=1)[:, :k].astype(np.int32, copy=False)

    return rank


def _camera_distance_difference(features, cam_labels, block_size=512):
    N = features.shape[0]
    features_t = features.T
    labels = np.asarray(cam_labels)
    cols = np.arange(N)
    same_sum = diff_sum = 0.0
    same_count = diff_count = 0

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        rows = np.arange(start, end)
        dist = 2 - 2 * np.matmul(features[start:end], features_t)
        upper = cols[None, :] > rows[:, None]
        cam_mask = labels[start:end, None] == labels[None, :]

        same = upper & cam_mask
        diff = upper & ~cam_mask
        same_sum += float(dist[same].sum())
        diff_sum += float(dist[diff].sum())
        same_count += int(same.sum())
        diff_count += int(diff.sum())

    if same_count == 0 or diff_count == 0:
        return 0.0
    return diff_sum / diff_count - same_sum / same_count


def _feature_distances(features, i, indices):
    return 2 - 2 * np.matmul(features[indices], features[i])


def _available_memory_bytes():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def _estimate_dense_jaccard_bytes(N, ckrnns, clqe):
    matrix = N * N
    dense_float_mats = 5
    dense_rank_mats = 3 if (ckrnns or clqe) else 1
    camera_masks = 2 if (ckrnns or clqe) else 1
    return matrix * (
        dense_float_mats * np.dtype(np.float32).itemsize
        + dense_rank_mats * np.dtype(np.int64).itemsize
        + camera_masks * np.dtype(bool).itemsize
    )


def _resolve_jaccard_memory_mode(requested, N, ckrnns, clqe):
    if requested in ("dense", "sparse"):
        return requested
    if requested != "auto":
        raise ValueError("jaccard_memory must be one of: auto, dense, sparse")

    available = _available_memory_bytes()
    estimate = _estimate_dense_jaccard_bytes(N, ckrnns, clqe)
    if available is not None and available >= estimate * 1.25:
        return "dense"
    return "sparse"


def _sparse_lqe(rows, neighbor_rank):
    qe_rows = []
    for neighbors in neighbor_rank:
        if len(neighbors) == 0:
            qe_rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue

        inds = np.concatenate([rows[n][0] for n in neighbors])
        vals = np.concatenate([rows[n][1] for n in neighbors]) / len(neighbors)
        uniq, inverse = np.unique(inds, return_inverse=True)
        weights = np.zeros(len(uniq), dtype=np.float32)
        np.add.at(weights, inverse, vals)
        qe_rows.append((uniq.astype(np.int32, copy=False), weights))
    return qe_rows


def _compute_sparse_neighbor_analysis(rows, cam_labels, pid_labels, N):
    cam_labels = np.asarray(cam_labels)
    pid_labels = np.asarray(pid_labels)
    if len(cam_labels) != N or len(pid_labels) != N:
        raise ValueError("cam_labels and pid_labels must have the same length as V")

    inter_props, inter_weights = [], []
    same_id_accs, same_id_weights = [], []

    for i, (inds, weights) in enumerate(rows):
        keep = inds != i
        inds = inds[keep]
        weights = weights[keep]
        if len(inds) == 0:
            continue

        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue

        inter_mask = cam_labels[inds] != cam_labels[i]
        same_id_mask = pid_labels[inds] == pid_labels[i]

        inter_props.append(float(inter_mask.mean()))
        inter_weights.append(float(weights[inter_mask].sum() / weight_sum))
        same_id_accs.append(float(same_id_mask.mean()))
        same_id_weights.append(float(weights[same_id_mask].sum() / weight_sum))

    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "avg_inter_camera_proportion": _mean(inter_props),
        "avg_inter_camera_weight": _mean(inter_weights),
        "avg_same_id_accuracy": _mean(same_id_accs),
        "avg_same_id_weight": _mean(same_id_weights),
    }


def v2jaccard_sparse(rows, N, mat_type, memmap_path=None):
    inv_inds = [[] for _ in range(N)]
    inv_vals = [[] for _ in range(N)]
    for row_idx, (inds, vals) in enumerate(rows):
        for col, val in zip(inds, vals):
            inv_inds[col].append(row_idx)
            inv_vals[col].append(val)

    for i in range(N):
        inv_inds[i] = np.asarray(inv_inds[i], dtype=np.int32)
        inv_vals[i] = np.asarray(inv_vals[i], dtype=mat_type)

    if memmap_path:
        dirname = osp.dirname(memmap_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        jaccard_dist = np.memmap(memmap_path, dtype=mat_type, mode="w+", shape=(N, N))
    else:
        jaccard_dist = np.empty((N, N), dtype=mat_type)
    for i, (ind_nonzero, vals_nonzero) in enumerate(rows):
        temp_min = np.zeros(N, dtype=mat_type)
        for ind, val in zip(ind_nonzero, vals_nonzero):
            images = inv_inds[ind]
            if len(images) == 0:
                continue
            temp_min[images] += np.minimum(val, inv_vals[ind])
        row = 1 - temp_min / (2 - temp_min)
        row[row < 0] = 0.0
        jaccard_dist[i] = row

    if memmap_path:
        jaccard_dist.flush()
    return jaccard_dist


def _distance_mode(ckrnns, clqe):
    if ckrnns and clqe:
        return "caj"
    if ckrnns and not clqe:
        return "ckrnns"
    if not ckrnns and clqe:
        return "clqe"
    return "baseline"


def _append_neighbor_analysis(row, fpath):
    dirname = osp.dirname(fpath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    fieldnames = [
        "dataset", "epoch", "mode", "ckrnns", "clqe",
        "k1", "k2", "k1_intra", "k1_inter", "k2_intra", "k2_inter",
        "num_samples", "avg_inter_camera_proportion",
        "avg_inter_camera_weight", "avg_same_id_accuracy",
        "avg_same_id_weight",
    ]
    write_header = not osp.exists(fpath) or osp.getsize(fpath) == 0
    with open(fpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _compute_neighbor_analysis(V, cam_labels, pid_labels):
    cam_labels = np.asarray(cam_labels)
    pid_labels = np.asarray(pid_labels)
    if len(cam_labels) != V.shape[0] or len(pid_labels) != V.shape[0]:
        raise ValueError("cam_labels and pid_labels must have the same length as V")

    inter_props, inter_weights = [], []
    same_id_accs, same_id_weights = [], []

    for i in range(V.shape[0]):
        inds = np.flatnonzero(V[i])
        inds = inds[inds != i]
        if len(inds) == 0:
            continue

        weights = V[i, inds]
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue

        inter_mask = cam_labels[inds] != cam_labels[i]
        same_id_mask = pid_labels[inds] == pid_labels[i]

        inter_props.append(float(inter_mask.mean()))
        inter_weights.append(float(weights[inter_mask].sum() / weight_sum))
        same_id_accs.append(float(same_id_mask.mean()))
        same_id_weights.append(float(weights[same_id_mask].sum() / weight_sum))

    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "avg_inter_camera_proportion": _mean(inter_props),
        "avg_inter_camera_weight": _mean(inter_weights),
        "avg_same_id_accuracy": _mean(same_id_accs),
        "avg_same_id_weight": _mean(same_id_weights),
    }


def _record_neighbor_analysis(args, epoch, ckrnns, clqe, k1, k2, k1_intra, k1_inter,
                              k2_intra, k2_inter, N, stats, neighbor_analysis_file):
    row = {
        "dataset": getattr(args, "dataset", ""),
        "epoch": epoch,
        "mode": _distance_mode(ckrnns, clqe),
        "ckrnns": int(ckrnns),
        "clqe": int(clqe),
        "k1": k1,
        "k2": k2,
        "k1_intra": k1_intra,
        "k1_inter": k1_inter,
        "k2_intra": k2_intra,
        "k2_inter": k2_inter,
        "num_samples": N,
    }
    row.update(stats)
    _append_neighbor_analysis(row, neighbor_analysis_file)
    print("Neighbor analysis:", stats)


def _compute_jaccard_distance_dense(features, cam_labels, epoch, args, pid_labels,
                                    neighbor_analysis_file, end):
    k1, k2 = args.k1, args.k2
    ckrnns, k1_intra, k1_inter = args.ckrnns, args.k1_intra, args.k1_inter
    clqe, k2_intra, k2_inter = args.clqe, args.k2_intra, args.k2_inter
    N = features.shape[0]
    mat_type = np.float32

    original_dist = 2 - 2 * np.matmul(features, features.T)

    cam_mask = (cam_labels.reshape(-1, 1) == cam_labels.reshape(1, -1))
    cam_diff = original_dist[np.triu(~cam_mask, k=1)].mean() - original_dist[np.triu(cam_mask, k=1)].mean()
    print('Camera difference: {:.2f}'.format(cam_diff))

    if ckrnns or clqe:
        inter_rank = np.argpartition(original_dist + 999.0 * cam_mask, range(k1_inter + 2))
        intra_rank = np.argpartition(original_dist + 999.0 * (~cam_mask), range(k1_intra + 2))
    global_rank = np.argpartition(original_dist, range(k1 + 2))

    if ckrnns:
        print(f"EPOCH[{epoch}] [CKRNNs] PARAMS: k1_intra: {k1_intra}, k1_inter: {k1_inter}")
    else:
        print(f"EPOCH[{epoch}] [KRNNs] PARAMS: k1: {k1}")

    if ckrnns:
        nn_inter = [k_reciprocal_neigh(inter_rank, i, k1_inter) for i in range(N)]
        nn_intra = [k_reciprocal_neigh(intra_rank, i, k1_intra) for i in range(N)]
        nn_k1 = [np.union1d(nn_intra[i], nn_inter[i]) for i in range(N)]
    else:
        nn_k1 = [k_reciprocal_neigh(global_rank, i, k1) for i in range(N)]
        nn_k1_half = [k_reciprocal_neigh(global_rank, i, int(np.around(k1 / 2))) for i in range(N)]

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index

        if not ckrnns:
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k1_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        dist = torch.from_numpy(original_dist[i][k_reciprocal_expansion_index]).unsqueeze(0)
        V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    if epoch == 0:
        print("Warm-up...")
        k2_intra, k2_inter = 3, 3
    if clqe:
        print(f"EPOCH[{epoch}] [CLQE] PARAMS: k2_intra: {k2_intra}, k2_inter: {k2_inter}")
    else:
        print(f"EPOCH[{epoch}] [LQE] PARAMS: k2: {k2}")
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            if clqe:
                k2nn = np.append(intra_rank[i, :k2_intra], inter_rank[i, :k2_inter])
            else:
                k2nn = global_rank[i, :k2]
            V_qe[i, :] = np.mean(V[k2nn, :], axis=0)
        V = V_qe

    if neighbor_analysis_file:
        if pid_labels is None:
            raise ValueError("pid_labels is required when neighbor_analysis_file is set")
        stats = _compute_neighbor_analysis(V, cam_labels, pid_labels)
        _record_neighbor_analysis(args, epoch, ckrnns, clqe, k1, k2, k1_intra, k1_inter,
                                  k2_intra, k2_inter, N, stats, neighbor_analysis_file)

    jaccard_dist = v2jaccard(V, N, mat_type)

    print("Distance computing time cost: {}".format(time.time() - end))
    return jaccard_dist


def compute_jaccard_distance(features=None, cam_labels=None, epoch=None, args=None,
                             pid_labels=None, neighbor_analysis_file=None):
    end = time.time()
    print('Computing CA-jaccard/jaccard distance...')
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(cam_labels, torch.Tensor):
        cam_labels = cam_labels.cpu().numpy()
    if isinstance(pid_labels, torch.Tensor):
        pid_labels = pid_labels.cpu().numpy()

    k1, k2 = args.k1, args.k2
    ckrnns, k1_intra, k1_inter = args.ckrnns, args.k1_intra, args.k1_inter
    clqe, k2_intra, k2_inter = args.clqe, args.k2_intra, args.k2_inter


    if ckrnns and clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (CKRNNS + CLQE)]"
    elif ckrnns and not clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (CKRNNS + LQE)]"
    elif not ckrnns and clqe:
        mode = f"EPOCH[{epoch}] [CAJaccard (KRNNS + CLQE)]"
    else:
        mode = f"EPOCH[{epoch}] [Jaccard (KRNNS + LQE)]"
    print(mode)

    features = np.asarray(features, dtype=np.float32)
    N = features.shape[0]
    mat_type = np.float32
    requested_memory_mode = getattr(args, "jaccard_memory", "auto")
    memory_mode = _resolve_jaccard_memory_mode(requested_memory_mode, N, ckrnns, clqe)
    print("Jaccard memory mode: {}{}".format(
        memory_mode, f" (requested: {requested_memory_mode})" if memory_mode != requested_memory_mode else ""))

    if memory_mode == "dense":
        return _compute_jaccard_distance_dense(features, cam_labels, epoch, args, pid_labels,
                                               neighbor_analysis_file, end)

    cam_diff = _camera_distance_difference(features, cam_labels)
    print('Camera difference: {:.2f}'.format(cam_diff))

    rank_cols = max(k1 + 2, int(np.around(k1 / 2)) + 2, k2, k1_intra + 2, k1_inter + 2, k2_intra, k2_inter)
    if ckrnns or clqe:
        inter_rank = _topk_rank_from_features(features, rank_cols, cam_labels=cam_labels, same_camera=False)
        intra_rank = _topk_rank_from_features(features, rank_cols, cam_labels=cam_labels, same_camera=True)
    global_rank = _topk_rank_from_features(features, rank_cols)

    ###################################
    #           KRNNs/CKRNNs          #
    ###################################
    if ckrnns:
        print(f"EPOCH[{epoch}] [CKRNNs] PARAMS: k1_intra: {k1_intra}, k1_inter: {k1_inter}")
    else:
        print(f"EPOCH[{epoch}] [KRNNs] PARAMS: k1: {k1}")

    if ckrnns:
        nn_inter = [k_reciprocal_neigh(inter_rank, i, k1_inter) for i in range(N)]
        nn_intra = [k_reciprocal_neigh(intra_rank, i, k1_intra) for i in range(N)]
        nn_k1 = [np.union1d(nn_intra[i], nn_inter[i]) for i in range(N)]
    else:
        nn_k1 = [k_reciprocal_neigh(global_rank, i, k1) for i in range(N)]
        nn_k1_half = [k_reciprocal_neigh(global_rank, i, int(np.around(k1 / 2))) for i in range(N)]

    rows = []
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index

        # Jaccard recall
        if not ckrnns:
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k1_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        ## element-wise unique
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        dist = _feature_distances(features, i, k_reciprocal_expansion_index)
        weight = np.exp(-dist + np.max(-dist))
        rows.append((k_reciprocal_expansion_index.astype(np.int32, copy=False),
                     (weight / np.sum(weight)).astype(mat_type, copy=False)))

    ################################
    #            LQE/CLQE          #
    ################################
    # warmup
    if epoch == 0:
        print("Warm-up...")
        k2_intra, k2_inter = 3, 3
    if clqe:
        print(f"EPOCH[{epoch}] [CLQE] PARAMS: k2_intra: {k2_intra}, k2_inter: {k2_inter}")
    else:
        print(f"EPOCH[{epoch}] [LQE] PARAMS: k2: {k2}")
    if k2 != 1:
        if clqe:
            neighbor_rank = [np.append(intra_rank[i, :k2_intra], inter_rank[i, :k2_inter]) for i in range(N)]
        else:
            neighbor_rank = [global_rank[i, :k2] for i in range(N)]
        rows = _sparse_lqe(rows, neighbor_rank)

    if neighbor_analysis_file:
        if pid_labels is None:
            raise ValueError("pid_labels is required when neighbor_analysis_file is set")
        stats = _compute_sparse_neighbor_analysis(rows, cam_labels, pid_labels, N)
        _record_neighbor_analysis(args, epoch, ckrnns, clqe, k1, k2, k1_intra, k1_inter,
                                  k2_intra, k2_inter, N, stats, neighbor_analysis_file)

    del features
    if "global_rank" in locals():
        del global_rank
    if "inter_rank" in locals():
        del inter_rank
    if "intra_rank" in locals():
        del intra_rank
    if "nn_k1" in locals():
        del nn_k1
    if "nn_k1_half" in locals():
        del nn_k1_half
    if "nn_inter" in locals():
        del nn_inter
    if "nn_intra" in locals():
        del nn_intra
    gc.collect()

    memmap_path = None
    if N >= 20000:
        memmap_path = osp.join(args.logs_dir, f"jaccard_epoch_{epoch}.mmap")
        print("Writing dense Jaccard distance to memmap: {}".format(memmap_path))

    jaccard_dist = v2jaccard_sparse(rows, N, mat_type, memmap_path=memmap_path)

    print("Distance computing time cost: {}".format(time.time() - end))
    return jaccard_dist


def v2jaccard(V, N, mat_type):
    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    return jaccard_dist
