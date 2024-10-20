import numpy as np

# # Define x and y with the given shapes
# x = np.random.rand(1, 512, 480, 480)
# y = np.random.rand(2, 512)

# # Perform matrix multiplication along the second axis
# result = np.einsum('ijkl,mj->imkl', x, y)

# # Output the shape of the result
# print(result.shape)

# import numpy as np

# # 生成随机数组
# x = np.random.rand(1, 3, 480, 480)

# # 比较第二个维度的大小并返回最大值对应的索引
# max_indices = np.argmax(x, axis=1)

# # 将索引保存在（480，480）的数组中
# index_array = np.zeros((480, 480))
# index_array.flatten()[np.argmax(x, axis=1).flatten()] = np.arange(3)  # 将索引0, 1, 2分配给相应的位置

# # 打印结果
# print(index_array)
import numpy as np

# 生成随机数组
x = np.random.rand(1, 3, 480, 480)

# 比较第二个维度的大小并返回最大值对应的索引
max_indices = np.argmax(x, axis=1).squeeze()

# 将索引保存在（480，480）的数组中
index_array = np.zeros((480, 480), dtype=int)
for i in range(480):
    for j in range(480):
        index_array[i, j] = max_indices[0, i, j]

# 打印结果
print(index_array)


# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import onnx
# import onnxruntime as ort
# import os
# import logging
# import onnxruntime
# import numpy as np
# from collections import OrderedDict

# # Define a simple model that applies a linear interpolation upsample
# class ResizeModel(torch.nn.Module):
#     def __init__(self):
#         super(ResizeModel, self).__init__()

#     def forward(self, x):
#         # Apply the linear (bilinear for 4D) upsampling
#         return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

# # Create an instance of the model
# model = ResizeModel()

# # Generate dummy input tensor
# torch.manual_seed(0)
# input_tensor = torch.randn(1, 3, 13,13)
# input_tensor_np=input_tensor.detach().numpy().astype(np.float32)
# input_tensor_np.tofile("test6_resize_input_1_3_13_13_fp32.bin")

# ## from local file
# # input_tensor_np=np.fromfile("test6_resize_input_1_3_13_13_fp32.bin",dtype=np.float32).reshape((1,3,13,13)).astype(np.float32)
# # input_tensor = torch.tensor(input_tensor_np)


# # Apply the model to generate the output
# output_tensor = model(input_tensor)
# output_tensor_np=output_tensor.detach().numpy().astype(np.float32)
# output_tensor_np.tofile("test6_resize_output_1_3_26_26_fp32_torch.bin")
# # Export the model to ONNX format
# torch.onnx.export(model,                     # model being run
#                   input_tensor,              # model input (or a tuple for multiple inputs)
#                   "resize_model_3.onnx",   # where to save the model
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=17,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],     # the model's input names
#                   output_names=['output'],   # the model's output names
#                   dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # variable length axes

# print(output_tensor.shape, output_tensor)


# def test_model_by_onnxruntime(model, save_dir, img):
#     logger.info("Test model by onnxruntime")
#     graph_out_names = [g_o.name for g_o in model.graph.output]
#     node_name = []
#     for node in model.graph.node:
#         for output in node.output:
#             a = onnx.ValueInfoProto(name=output)
#             model.graph.output.extend([onnx.ValueInfoProto(name=output)])
#             if node.name:
#                 node_name.append(node.name)
#             else:
#                 output_name = node.output[0]
#                 node_name.append(str(node.op_type) + "_" + output_name)

#     ort_session = onnxruntime.InferenceSession(model.SerializeToString(),
#                                      providers=['CPUExecutionProvider'])

#     ort_inputs = {}
#     #TODO if >1 input.
#     if (len(ort_session.get_inputs())> 1):
#         for i, input_ele in enumerate(ort_session.get_inputs()):
#             np_file = "./" + input_ele.name + ".npy"
#             img = np.load(np_file)
#             img = np.expand_dims(img,0).astype("float32")
#             img.tofile(output_dir + f"data_{str(i)}.bin")
#             ort_inputs[input_ele.name] = img
#     else:
#         for i, input_ele in enumerate(ort_session.get_inputs()):
#             ort_inputs[input_ele.name] = img

#     outputs = [x.name for x in ort_session.get_outputs()]
#     ort_outs = ort_session.run(outputs, ort_inputs)
#     outs_dict = OrderedDict(zip(outputs, ort_outs))
#     output_num = len(ort_outs) - len(node_name)
#     ort_outs = ort_outs[output_num::]
#     for i in range(len(ort_outs)):
#         ort_outs_1 = ort_outs[i].astype(np.float32)
#         # np.save(save_dir + node_name[i].replace("/","_") + ".npy", ort_outs_1)
#         ort_outs_1.tofile("test6_resize_output_1_32_26_26_fp32_onnxruntime.bin")

#     # out_list = {out.name:out for out in model.graph.outputs}
#     out_tensor = [outs_dict[outname].astype(np.float32) for outname in graph_out_names]
#     logger.info("Test model by onnxruntime success")
#     return tuple(out_tensor),graph_out_names
# logger = logging.getLogger("[ONNXOPTIMIZER]")
# output_dir='./'
# new_model_file = "./resize_model_2.onnx"

# onnx_model = onnx.load(new_model_file)
# # img=np.fromfile("fp_Resize_Input_hansen.bin",dtype=np.float32).reshape((1,3,224,56)).astype(np.float32)
# # img.tofile("data_input_fp32.bin")

# img=np.fromfile("test6_resize_input_1_3_13_13_fp32.bin",dtype=np.float32).reshape((1,3,13,13)).astype(np.float32)
# img.tofile("data_input_fp32_13_13.bin")


# outputs,outputs_name = test_model_by_onnxruntime(onnx_model, "./", img)


