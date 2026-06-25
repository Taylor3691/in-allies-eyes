from __future__ import print_function, absolute_import
import argparse
import gc
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

src_root = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'src')
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from caj import datasets
from caj import models
from caj.models.cm import ClusterMemory
from caj.trainers import ClusterContrastTrainer
from caj.evaluators import Evaluator, extract_features
from caj.utils.data import IterLoader
from caj.utils.data import transforms as T
from caj.utils.data.preprocessor import Preprocessor
from caj.utils.logging import Logger
from caj.utils.serialization import load_checkpoint, save_checkpoint
from caj.utils.caj_rerank import compute_jaccard_distance
from caj.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    total_samples = len(sampler) if sampler is not None else len(train_set)
    drop_last = True
    if total_samples < batch_size:
        drop_last = False
        print("Warning: train set size ({}) is smaller than batch_size ({}). "
              "Setting drop_last=False to prevent empty dataloader.".format(total_samples, batch_size))

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=drop_last), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model_kwargs = dict(
        num_features=args.features,
        norm=True,
        dropout=args.dropout,
        num_classes=0,
        pooling_type=args.pooling_type,
    )
    if args.pretrain_path and args.arch in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
        model_kwargs['pretrained_path'] = args.pretrain_path
    model = models.create(args.arch, **model_kwargs)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # generate camera labels
    if args.dataset == 'msmt17':
        cam_labels = np.array([cid - 1 for _, _, cid in sorted(dataset.train)])
    else:
        cam_labels = np.array([cid for _, _, cid in sorted(dataset.train)])
    pid_labels = np.array([pid for _, pid, _ in sorted(dataset.train)])
    neighbor_analysis_file = None
    if args.neighbor_analysis:
        neighbor_analysis_file = args.neighbor_analysis_file
        if not neighbor_analysis_file:
            neighbor_analysis_file = osp.join(args.logs_dir, 'neighbor_analysis.csv')
        print('Neighbor analysis file: {}'.format(neighbor_analysis_file))

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, cam_labels=cam_labels, epoch=epoch, args=args,
                                                   pid_labels=pid_labels,
                                                   neighbor_analysis_file=neighbor_analysis_file)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            del rerank_dist
            gc.collect()
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)

        del cluster_loader, features

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        num_outlier = len(dataset.train) - len(pseudo_labeled_dataset)
        print(
            '==> Statistics for epoch {}: {} clusters, {} un-cluster instances'.format(epoch, num_cluster, num_outlier))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1) or (epoch >= 40):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CA-Jaccard: Camera-aware Jaccard Distance for Person Re-identification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--resume', type=str, default='')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=4)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    parser.add_argument('--pretrain-path', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'pretrained_models', 'resnet50-11ad3fa6.pth'),
                        help='local ImageNet pretrained ResNet checkpoint; set empty to use torchvision defaults')
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    parser.add_argument('--pooling-type', type=str, default='avg')
    parser.add_argument('--no-cam', action="store_true")
    parser.add_argument('--neighbor-analysis', action='store_true',
                        help='write per-epoch neighbor-analysis metrics for Fig. 3')
    parser.add_argument('--neighbor-analysis-file', type=str, default='',
                        help='CSV path for neighbor-analysis metrics; defaults to logs_dir/neighbor_analysis.csv')

    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")

    # Jaccard
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--jaccard-memory', choices=['auto', 'dense', 'sparse'], default='auto',
                        help="memory strategy for clustering Jaccard distance")

    # CKRNNs
    parser.add_argument('--ckrnns', action='store_true')
    parser.add_argument('--k1-intra', type=int, default=5)
    parser.add_argument('--k1-inter', type=int, default=20)

    # CLQE
    parser.add_argument('--clqe', action='store_true')
    parser.add_argument('--k2-intra', type=int, default=3)
    parser.add_argument('--k2-inter', type=int, default=3)
    main()
