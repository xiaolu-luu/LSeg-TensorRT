import os
import logging
import onnxruntime
import numpy as np
import torch
import onnx
import gdown
from collections import OrderedDict
import torchvision.transforms as transforms
from pathlib import Path
from lseg.modules.models.lseg_net import LSegEncNetExp


def test_model_by_onnxruntime(model, save_dir, img):
    logger.info("Test model by onnxruntime")
    graph_out_names = [g_o.name for g_o in model.graph.output]
    node_name = []
    for node in model.graph.node:
        for output in node.output:
            a = onnx.ValueInfoProto(name=output)
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            if node.name:
                node_name.append(node.name)
            else:
                output_name = node.output[0]
                node_name.append(str(node.op_type) + "_" + output_name)

    ort_session = onnxruntime.InferenceSession(model.SerializeToString(),
                                     providers=['CPUExecutionProvider'])

    ort_inputs = {}
    #TODO if >1 input.
    if (len(ort_session.get_inputs())> 1):
        for i, input_ele in enumerate(ort_session.get_inputs()):
            np_file = "./" + input_ele.name + ".npy"
            img = np.load(np_file)
            img = np.expand_dims(img,0).astype("float32")
            img.tofile(output_dir + f"data_{str(i)}.bin")
            ort_inputs[input_ele.name] = img
    else:
        for i, input_ele in enumerate(ort_session.get_inputs()):
            ort_inputs[input_ele.name] = img

    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)
    outs_dict = OrderedDict(zip(outputs, ort_outs))
    output_num = len(ort_outs) - len(node_name)
    ort_outs = ort_outs[output_num::]
    for i in range(len(ort_outs)):
        ort_outs_1 = ort_outs[i].astype(np.float32)
        # np.save(save_dir + node_name[i].replace("/","_") + ".npy", ort_outs_1)
        ort_outs_1.tofile("./data/output_1_512_480_480_fp32_onnxruntime.bin")

    # out_list = {out.name:out for out in model.graph.outputs}
    out_tensor = [outs_dict[outname].astype(np.float32) for outname in graph_out_names]
    logger.info("Test model by onnxruntime success")
    return tuple(out_tensor),graph_out_names


def _init_lseg():
        crop_size = 480  # 480
        base_size = 520  # 520
        if torch.cuda.is_available():
            device = "cpu"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        lseg_model = LSegEncNetExp("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[1] /"vlmaps_lseg"/"vlmaps" /"lseg" / "checkpoints"
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

# Generate dummy input tensor
torch.manual_seed(0)
input_tensor = torch.randn(1, 3, 480,480)
input_tensor_np=input_tensor.detach().numpy().astype(np.float32)
input_tensor_np.tofile("./data/input_1_3_480_480_fp32.bin")

# Create the onnxruntime
logger = logging.getLogger("[ONNXOPTIMIZER]")
output_dir='./'
new_model_file = "./onnx/lseg.onnx"

onnx_model = onnx.load(new_model_file)

img=np.fromfile("./data/input_1_3_480_480_fp32.bin",dtype=np.float32).reshape((1,3,480,480)).astype(np.float32)
# img.tofile("data_input_fp32_13_13.bin")
print(img)
print("======================")
outputs,outputs_name = test_model_by_onnxruntime(onnx_model, "./", img)
print("outputs",outputs[0])
print("output.shape",outputs[0].shape)


# init lseg model
lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = _init_lseg()

if torch.cuda.is_available():
    device = "cpu"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

input_tensor = torch.from_numpy(img)


input_tensor = input_tensor.float()
lseg_model.eval()
with torch.no_grad():
    output = lseg_model(input_tensor)
print(output)    
