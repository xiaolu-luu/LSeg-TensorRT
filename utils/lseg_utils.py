import math

import numpy as np
import cv2
import torch

from matplotlib import pyplot as plt

from utils.visual_utils import *

from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image
import onnx
import onnxsim
import time 
def export_norm_onnx(model, file, input1):
    torch.onnx.export(
        model         = model, 
        args          = input1,
        f             = file,
        input_names   = ["input"],
        output_names  = ["output"],
        opset_version = 17,
        )#dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}}

    print("Finished normal onnx export")

    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    # 使用onnx-simplifier来进行onnx的简化。
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)
    
def get_lseg_feat(
    model: LSegEncNet,
    image: np.array,
    labels,
    transform,
    device,
    crop_size=480,
    base_size=520,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.5, 0.5, 0.5],
    vis=False,
):
    vis_image = image.copy()
    image = transform(image).unsqueeze(0).to(device) # torch.Size([1, 3, 480, 480])
    img = image[0].permute(1, 2, 0) # torch.Size([480, 480, 3])
    img = img * 0.5 + 0.5 # torch.Size([480, 480, 3])

    batch, _, h, w = image.size()
    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)

    # long_size = int(math.ceil(base_size * scale))
    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{"mode": "bilinear", "align_corners": True})#shape:torch.Size([1, 3, 347, 520])

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        print(pad_img.shape)
        with torch.no_grad():
            # outputs = model(pad_img)
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _, _, ph, pw = pad_img.shape  # .size() shape:torch.Size([1, 3, 480, 520])
        assert ph >= height and pw >= width
        h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c, ph, pw).zero_().to(device)
                logits_outputs = image.new().resize_(batch, len(labels), ph, pw).zero_().to(device)
            count_norm = image.new().resize_(batch, 1, ph, pw).zero_().to(device)
        # grid evaluation
        model.eval()
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                # totaltime = 0
                # start_time = time.time()
                # for i in range (100):
                with torch.no_grad():
                        # output = model(pad_crop_img)
                        # export_norm_onnx(model, "./lseg2_ns.onnx", pad_crop_img,labels)
                    output, logits = model(pad_crop_img, labels)
                        # totaltime+=(time.time() - start_time)*1000
                # print(totaltime)
                cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                cropped_logits = crop_image(logits, 0, h1 - h0, 0, w1 - w0)
                outputs[:, :, h0:h1, w0:w1] += cropped
                logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                count_norm[:, :, h0:h1, w0:w1] += 1
        assert (count_norm == 0).sum() == 0
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:, :, :height, :width]
        logits_outputs = logits_outputs[:, :, :height, :width]
    outputs = outputs.cpu()
    outputs = outputs.numpy()  # B, D, H, W
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]
    if vis:
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        seg = mask.convert("RGBA")
        # cv2.imshow("image", vis_image[:, :, [2, 1, 0]])
        # cv2.waitKey()
        fig = plt.figure()
        plt.imshow(seg)
        plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 10})
        plt.axis("off")

        plt.tight_layout()
        # plt.show()
        plt.savefig("seg_image_pt.png")  # 定义保存路径和文件名
        plt.close(fig)  # 关闭图形，避免显示

    return outputs
