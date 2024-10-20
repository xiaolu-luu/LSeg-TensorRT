# Judge whether a ONNX file is supported by TensorRT natively
# polygraphy inspect capability lseg1-opset17.onnx \
#     > ./log/result-06-B.log 2>&1
#     --trt-min-shapes 'input:[1,3, 480,480]' \
#     --trt-opt-shapes 'input:[1,3, 480,480]' \
#     --trt-max-shapes 'input:[1,3, 480,480]' \



# Compare the output between Onnxruntime and TensorRT
# polygraphy run ./lseg_gs2.onnx \
#     --onnxrt --trt \
#     --save-engine=./engines/clip_textual_poly.plan \
#     --onnx-outputs output\
#     --trt-outputs  output\
    # --trt-min-shapes 'input:[1,3,480,480]' \
    # --trt-opt-shapes 'input:[1,3,480,480]' \
    # --trt-max-shapes 'input:[1,3,480,480]' \
#     --input-shapes   'input:[1,3, 480,480]' \
#     --atol 1e-3 --rtol 1e-3 \
#     --verbose \
#     > ./log/result-poly-5.log 2>&1




# polygraphy run ./onnx/lseg2_ns.onnx \
#     --onnxrt --trt \
#     --save-engine=./engines/lseg_poly.plan \
#     --pool-limit workspace:5G \
#     --onnx-outputs /blocks.0/attn/MatMul_output_0 \
#     --trt-outputs /blocks.0/attn/MatMul_output_0 \
#     --input-shapes   'input:[1,3,480,480]' \
#     --atol 1e-3 --rtol 1e-3 \
#     --verbose \
#     > ./log/result-poly-9.log 2>&1

polygraphy run ./onnx/lseg.onnx \
    --onnxrt --trt \
    --save-engine=./engines/lseg_poly.plan \
    --pool-limit workspace:5G \
    --onnx-outputs output \
    --trt-outputs output \
    --input-shapes   'input:[1,3,480,480]' \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    > ./log/result-poly.log 2>&1

