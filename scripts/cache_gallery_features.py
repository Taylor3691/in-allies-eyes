#!/usr/bin/env python3
import os
import os.path as osp
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to python path
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..', 'src'))

from caj import datasets
from caj import models
from caj.utils.data import transforms as T
from caj.utils.data.preprocessor import Preprocessor
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
    from collections import OrderedDict
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


REPO_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))


def main():
    parser = argparse.ArgumentParser(description="Precompute and Cache Market-1501 Gallery Embeddings")
    parser.add_argument('--data-dir', type=str, default=osp.join(REPO_ROOT, 'data'))
    parser.add_argument('--resume', type=str, default=osp.join(REPO_ROOT, 'pretrained_models', 'market_resnet50_model_120_rank1_945.pth'))
    parser.add_argument('--checkpoint-format', type=str, default='auto')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--bot-neck-feat', type=str, default='after')
    parser.add_argument('--bot-no-feat-norm', action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=str, default=osp.join(REPO_ROOT, 'market1501_gallery_features.npy'))
    args = parser.parse_args()

    # Create dataset
    root = osp.join(args.data_dir, 'market1501')
    dataset = datasets.create('market1501', root)

    # Setup transformer
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    # Gallery loader
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True
    )

    # Load checkpoint
    checkpoint = load_checkpoint(args.resume)
    raw_state_dict = checkpoint['state_dict']
    checkpoint_format = args.checkpoint_format
    if checkpoint_format == 'auto' and is_bot_state_dict(raw_state_dict):
        checkpoint_format = 'bot'

    # Build model
    if checkpoint_format == 'bot':
        num_classes = getattr(dataset, 'num_train_pids', 0)
        model = build_bot_model(args, num_classes)
    else:
        model = models.create(args.arch, pretrained=False, num_features=0, dropout=0, num_classes=0)

    # Load weights
    if checkpoint_format == 'bot':
        state_dict = raw_state_dict
    else:
        state_dict = maybe_convert_state_dict(raw_state_dict, checkpoint_format)
    
    copy_state_dict(state_dict, model.model if checkpoint_format == 'bot' else model, strip='module.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Extract features
    gallery_features = []
    gallery_paths = []
    gallery_pids = []
    gallery_camids = []

    print("==> Extracting gallery features...")
    with torch.no_grad():
        for i, (imgs, fnames, pids, camids, _) in enumerate(gallery_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            gallery_features.append(outputs.cpu().numpy())
            gallery_paths.extend(fnames)
            gallery_pids.extend(pids.numpy())
            gallery_camids.extend(camids.numpy())

            if (i + 1) % 10 == 0:
                print(f"Progress: [{i+1}/{len(gallery_loader)}]")

    gallery_features = np.concatenate(gallery_features, axis=0)

    # Save to npy dict
    data_dict = {
        'features': gallery_features,
        'image_paths': gallery_paths,
        'pids': np.array(gallery_pids),
        'camids': np.array(gallery_camids)
    }

    # Make output directory if needed
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    np.save(args.output, data_dict)
    print(f"==> Successfully cached {len(gallery_paths)} gallery embeddings to {args.output}")

if __name__ == '__main__':
    main()
