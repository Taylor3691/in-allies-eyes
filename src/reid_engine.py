import os
import sys
import threading
import time

import torch
import torch.nn as nn


# Global Re-ID Model variables for lazy loading
reid_model: tuple[nn.Module, torch.device] | None = None
reid_model_lock = threading.Lock()
_model_loading_thread = None
_model_loading_error = None


def _bg_load_model():
    global reid_model, _model_loading_error
    try:
        import torch.nn.functional as F

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

        bot_root = os.path.join(os.path.dirname(__file__), '..', 'thirdparty', 'bot')
        if bot_root not in sys.path:
            sys.path.insert(0, bot_root)
        from modeling.baseline import Baseline

        # Market-1501 dataset has 751 train pids
        num_classes = 751
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

        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained_models', 'market_resnet50_model_120_rank1_945.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run scripts/download_pretrained_models.py demo.")

        checkpoint = load_checkpoint(checkpoint_path)
        raw_state_dict = checkpoint['state_dict']

        copy_state_dict(raw_state_dict, model.model, strip='module.')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        reid_model = (model, device)
        print("==> Re-ID model loaded successfully in background thread.")
    except Exception as e:
        _model_loading_error = e
        import traceback
        traceback.print_exc()


def get_reid_model():
    global reid_model, _model_loading_thread, _model_loading_error
    if reid_model is not None:
        return reid_model

    with reid_model_lock:
        if reid_model is None:
            if _model_loading_thread is None:
                print("==> Starting Re-ID model lazy load in background thread...")
                _model_loading_error = None
                _model_loading_thread = threading.Thread(target=_bg_load_model)
                _model_loading_thread.start()

            # Yield control periodically to allow event loops to run
            while _model_loading_thread.is_alive():
                time.sleep(0.05)

            _model_loading_thread = None
            if _model_loading_error is not None:
                raise _model_loading_error

    assert reid_model is not None
    return reid_model
