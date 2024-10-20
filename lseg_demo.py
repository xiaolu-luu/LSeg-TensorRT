import os
import gdown

import cv2
import torch
import torchvision.transforms as transforms
import onnx
import onnxsim
from pathlib import Path
from utils.lseg_utils import get_lseg_feat
from lseg.modules.models.lseg_net import LSegEncNet

def _init_lseg():
        crop_size = 480  # 480
        base_size = 520  # 520
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
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
rgb_path = "./chair.jpg"
bgr = cv2.imread(str(rgb_path))
# If you don't want to change the scale of the image, you can comment the following line
bgr = cv2.resize(bgr, (480,480))


rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"




pix_feats = get_lseg_feat(
                lseg_model, rgb, ["chair","other"], lseg_transform, device, crop_size, base_size, norm_mean, norm_std ,vis=True)