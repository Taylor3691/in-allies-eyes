import os
import sys
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global dict of loaded models mapping dataset key to (model, device)
reid_models = {}
reid_model_lock = threading.Lock()


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


def get_reid_model(dataset_name="Market1501"):
    global reid_models

    # Normalize dataset name key
    key = "cuhk03" if dataset_name == "CUHK03" else "market1501"

    if key in reid_models:
        return reid_models[key]

    with reid_model_lock:
        if key not in reid_models:
            print(f"==> Loading Re-ID model for {key}...")

            from ..caj.utils.serialization import load_checkpoint, copy_state_dict

            if REPO_ROOT not in sys.path:
                sys.path.insert(0, REPO_ROOT)
            from thirdparty.bot.modeling.baseline import Baseline

            # Define dataset-specific checkpoint filename
            checkpoint_filename = (
                'cuhk03_resnet50_model_120_rank1_608.pth'
                if key == "cuhk03"
                else 'market_resnet50_model_120_rank1_945.pth'
            )

            checkpoint_path = os.path.join(REPO_ROOT, 'pretrained_models', checkpoint_filename)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run scripts/download_pretrained_models.py demo.")

            checkpoint = load_checkpoint(checkpoint_path)
            raw_state_dict = checkpoint['state_dict']

            # Resolve num_classes dynamically from classifier weight shape
            classifier_key = None
            for k in raw_state_dict.keys():
                if k.endswith('classifier.weight'):
                    classifier_key = k
                    break

            if classifier_key is None:
                raise KeyError(f"Could not find classifier.weight in the state dict of {checkpoint_path}")

            num_classes = raw_state_dict[classifier_key].shape[0]
            print(f"==> Dynamically resolved num_classes = {num_classes} from checkpoint.")

            raw_model = Baseline(
                num_classes=num_classes,
                last_stride=1,
                model_path='',
                neck='bnneck',
                neck_feat='after',
                model_name='resnet50',
                pretrain_choice='none',
            )
            model = NormalizedFeatureModel(raw_model, normalize=True)

            copy_state_dict(raw_state_dict, model.model, strip='module.')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()

            reid_models[key] = (model, device)
            print(f"==> Re-ID model for {key} loaded successfully.")

    return reid_models[key]
