trtexec --onnx=./lseg1-opset17.onnx \
        --saveEngine=./engines/lseg.engine \
        --profilingVerbosity=detailed \
        --dumpOutput \
        --dumpProfile \
        --dumpLayerInfo \
        --exportOutput=./log/build_output_lseg.log \
        --exportProfile=./log/build_profile_lseg.log \
        --exportLayerInfo=./log/build_layer_info_lseg.log \
        --warmUp=200 \
        --iterations=50 \
        --verbose \
        > ./log/build_lseg.log

# Run trtexec from ONNX file without any more option
# trtexec \
#     --onnx=lseg1-opset17.onnx \
#     > ./log/result-01.log 2>&1

