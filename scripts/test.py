from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

src_root = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'src')
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from caj import datasets
from caj import models
from caj.models.dsbn import convert_dsbn, convert_bn
from caj.evaluators import Evaluator
from caj.utils.data import transforms as T
from caj.utils.data.preprocessor import Preprocessor
from caj.utils.logging import Logger
from caj.utils.serialization import load_checkpoint, copy_state_dict


class NormalizedFeatureModel(nn.Module):
    def __init__(self, model, normalize=True):
        super(NormalizedFeatureModel, self).__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, inputs):
        outputs = self.model(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if self.normalize:
            outputs = F.normalize(outputs, dim=1, p=2)
        return outputs


def convert_bot_state_dict(state_dict):
    converted = OrderedDict()
    layer_map = {
        'base.conv1.': 'base.0.',
        'base.bn1.': 'base.1.',
        'base.layer1.': 'base.4.',
        'base.layer2.': 'base.5.',
        'base.layer3.': 'base.6.',
        'base.layer4.': 'base.7.',
        'bottleneck.': 'feat_bn.',
    }

    for name, param in state_dict.items():
        if name.startswith('classifier.'):
            continue
        new_name = name
        for old_prefix, new_prefix in layer_map.items():
            if name.startswith(old_prefix):
                new_name = new_prefix + name[len(old_prefix):]
                break
        converted[new_name] = param

    return converted


def maybe_convert_state_dict(state_dict, checkpoint_format):
    if checkpoint_format == 'bot':
        return convert_bot_state_dict(state_dict)
    if checkpoint_format == 'auto':
        if any(name.startswith('base.conv1.') for name in state_dict.keys()):
            print('==> Detected BoT checkpoint format; converting keys for CAJ ResNet')
            return convert_bot_state_dict(state_dict)
    return state_dict


def is_bot_state_dict(state_dict):
    return any(name.startswith('base.conv1.') for name in state_dict.keys())


def build_bot_model(args, num_classes):
    if args.arch != 'resnet50':
        raise ValueError("BoT checkpoint support currently only supports --arch resnet50")

    bot_root = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'src', 'thirdparty', 'bot')
    if bot_root not in sys.path:
        sys.path.insert(0, bot_root)

    from modeling.baseline import Baseline

    print("==> Building BoT ResNet50 BNNeck model")
    model = Baseline(
        num_classes=num_classes,
        last_stride=1,
        model_path='',
        neck='bnneck',
        neck_feat=args.bot_neck_feat,
        model_name='resnet50',
        pretrain_choice='none',
    )
    return NormalizedFeatureModel(model, normalize=(not args.bot_no_feat_norm))


def build_caj_model(args):
    return models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout,
                         num_classes=0, pooling_type=args.pooling_type)


def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    log_dir = args.logs_dir if args.logs_dir else osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset, test_loader = get_data(args.dataset, args.data_dir, args.height,
                                    args.width, args.batch_size, args.workers)

    # Load from checkpoint before model construction so auto format detection can
    # instantiate the matching architecture.
    checkpoint = load_checkpoint(args.resume)
    raw_state_dict = checkpoint['state_dict']
    checkpoint_format = args.checkpoint_format
    if checkpoint_format == 'auto' and is_bot_state_dict(raw_state_dict):
        checkpoint_format = 'bot'
        print('==> Detected BoT checkpoint format')

    # Create model
    if checkpoint_format == 'bot':
        num_classes = getattr(dataset, 'num_train_pids', 0)
        model = build_bot_model(args, num_classes)
    else:
        model = build_caj_model(args)

    if args.dsbn and checkpoint_format == 'bot':
        raise ValueError("--dsbn is only supported for CAJ-format models")
    if args.dsbn:
        print("==> Load the model with domain-specific BNs")
        convert_dsbn(model)

    if checkpoint_format == 'bot':
        state_dict = raw_state_dict
    else:
        state_dict = maybe_convert_state_dict(raw_state_dict, checkpoint_format)
    copy_state_dict(state_dict, model.model if checkpoint_format == 'bot' else model, strip='module.')

    if args.dsbn:
        print("==> Test with {}-domain BNs".format("source" if args.test_source else "target"))
        convert_bn(model, use_target=(not args.test_source))

    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    model.eval()
    evaluator = Evaluator(model)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, args, cmc_flag=True, rerank=args.rerank)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--resume', type=str,
                        default="pretrained_models/",
                        metavar='PATH')
    parser.add_argument('--checkpoint-format', type=str, default='auto',
                        choices=['auto', 'caj', 'bot'],
                        help='checkpoint key format')
    # testing configs
    parser.add_argument('--dsbn', action='store_true',
                        help="test on the model with domain-specific BN")
    parser.add_argument('--test-source', action='store_true',
                        help="test on the source domain")
    parser.add_argument('--seed', type=int, default=1)
    working_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default='',
                        help='directory for test log; defaults to checkpoint directory')
    parser.add_argument('--pooling-type', type=str, default='avg')
    parser.add_argument('--bot-neck-feat', type=str, default='after',
                        choices=['after', 'before'],
                        help='BoT BNNeck feature to use during testing')
    parser.add_argument('--bot-no-feat-norm', action='store_true',
                        help='disable L2 normalization for BoT test features')
    parser.add_argument('--embedding_features_path', type=str,
                        default='')

    # rerank setting
    parser.add_argument('--rerank', action='store_true')
    # Jaccard
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # CKRNNs
    parser.add_argument('--ckrnns', action='store_true')
    parser.add_argument('--k1-intra', type=int, default=5)
    parser.add_argument('--k1-inter', type=int, default=20)

    # CLQE
    parser.add_argument('--clqe', action='store_true')
    parser.add_argument('--k2-intra', type=int, default=2)
    parser.add_argument('--k2-inter', type=int, default=4)
    main()
