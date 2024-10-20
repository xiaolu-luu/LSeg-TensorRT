
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxsim
# tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
# tensor1 = gs.Variable("tensor1", np.float32, None)
# tensor2 = gs.Variable("tensor2", np.float32, None)
# tensor3 = gs.Variable("tensor3", np.float32, None)
# constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))

# node0 = gs.Node("Identity", "myIdentity0", inputs=[tensor0], outputs=[tensor1])
# node1 = gs.Node("Add", "myAdd", inputs=[tensor1, constant0], outputs=[tensor2])
# node2 = gs.Node("Identity", "myIdentity1", inputs=[tensor2], outputs=[tensor3])

# graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])
# graph.cleanup().toposort()
# onnx.save(gs.export_onnx(graph), "model-04-01.onnx")

# del graph

# replace node by edit the operator type
graph = gs.import_onnx(onnx.load("./onnx/lseg.onnx"))  # load the graph from ONNX file
i=0
for node in graph.nodes:
    if node.op == "Resize"and node.name == "/refinenet1/Resize" : 
        newNode = gs.Node("Resize", "myResize{}".format(i), inputs=node.inputs, outputs=node.outputs)
        newNode.attrs["coordinate_transformation_mode"]="align_corners"
        newNode.attrs["cubic_coeff_a"]=-0.75
        newNode.attrs["mode"]="linear"
        newNode.attrs["nearest_mode"]="floor"
        graph.nodes.append(newNode)
        node.outputs = []
        i+=1
        print(i)
        # node.name = "mySub"  # it's OK to change the name of the node or not

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./onnx/lseg_gs.onnx")
model_onnx = onnx.load("./onnx/lseg_gs.onnx")

# 检查导入的onnx model
onnx.checker.check_model(model_onnx)

# 使用onnx-simplifier来进行onnx的简化。
print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, "assert check failed"
onnx.save(model_onnx, "./onnx/lseg_gs1.onnx")

