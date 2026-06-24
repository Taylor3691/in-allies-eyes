# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import sys
import time
import torch
import types

if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.string_classes = (str, bytes)
    sys.modules["torch._six"] = torch_six

from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


def eval_func_chunked(qf, gf, q_pids, g_pids, q_camids, g_camids, max_rank=50, query_chunk_size=1024):
    num_q = qf.shape[0]
    num_g = gf.shape[0]
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    gf_sq = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()
    total_chunks = (num_q + query_chunk_size - 1) // query_chunk_size
    start_time = time.time()
    for start in range(0, num_q, query_chunk_size):
        end = min(start + query_chunk_size, num_q)
        chunk_idx = start // query_chunk_size + 1
        qf_chunk = qf[start:end]
        distmat = torch.pow(qf_chunk, 2).sum(dim=1, keepdim=True) + gf_sq
        distmat.addmm_(qf_chunk, gf.t(), beta=1, alpha=-2)
        indices = np.argsort(distmat.cpu().numpy(), axis=1)
        matches = (g_pids[indices] == q_pids[start:end, np.newaxis]).astype(np.int32)

        for local_idx in range(end - start):
            q_idx = start + local_idx
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            order = indices[local_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            orig_cmc = matches[local_idx][keep]
            if not np.any(orig_cmc):
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        if chunk_idx == 1 or chunk_idx % 10 == 0 or end == num_q:
            elapsed = time.time() - start_time
            print("Eval chunk {}/{} ({}/{}) elapsed {:.1f}s".format(
                chunk_idx, total_chunks, end, num_q, elapsed))

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        if m * n > 2000000000:
            print("Computing eval metrics in query chunks for {}x{} distance matrix".format(m, n))
            cmc, mAP = eval_func_chunked(qf, gf, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)
        else:
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
            distmat = distmat.cpu().numpy()
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
