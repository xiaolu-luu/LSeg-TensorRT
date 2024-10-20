import os
import gdown
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import onnx
import onnxsim
from pathlib import Path
from utils.lseg_utils import export_norm_onnx
from lseg.modules.models.lseg_net import LSegEncNetExp

def _init_lseg():
        crop_size = 480  # 480
        base_size = 520  # 520
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        lseg_model = LSegEncNetExp("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[0] /"lseg" / "checkpoints"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(checkpoint_path, map_location=device)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

# init lseg model
lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = _init_lseg()


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
torch.manual_seed(0)
dummy_image = torch.randn(1, 3, 480,480).to(device)
# dummy_image_np=dummy_image.detach().numpy().astype(np.float32)


lseg_model.eval()
with torch.no_grad():
    output = lseg_model(dummy_image)
    export_norm_onnx(lseg_model, "./onnx/lseg.onnx", dummy_image)
    
print(output.shape)
print("Finish")    