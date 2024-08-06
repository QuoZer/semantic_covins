cd /workspace/nanosam && \
    /usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10

cd /workspace/nanosam/data/ && \
    gdown https://drive.google.com/uc?id=14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR && \
    ls -lh && \
    cd /workspace/nanosam/ && \
    trtexec \
        --onnx=data/resnet18_image_encoder.onnx \
        --saveEngine=data/resnet18_image_encoder.engine \
        --fp16    