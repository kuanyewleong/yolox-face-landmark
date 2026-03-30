# YOLOX-L Face Detection + 5-Point Landmark Training

This repo trains a YOLOX-L-based face detector with an additional 5-point landmark branch.

## Training strategy

- Phase 1: detection pretrain on WIDER FACE only
- Phase 2: multi-task fine-tune on mixed WIDER FACE + WFLW

## Repository tree
yolox-face-landmark/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   └── yolox_l_face_landmark.py
├── tools/
│   ├── train.py
│   ├── eval.py
│   ├── export_onnx.py
│   ├── build_trt.py
│   └── infer_onnx.py
├── yolox_face/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── env.py
│   │   ├── box_ops.py
│   │   ├── checkpoint.py
│   │   ├── ema.py
│   │   ├── lr_scheduler.py
│   │   └── postprocess.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   ├── collate.py
│   │   ├── wider_face.py
│   │   ├── wflw.py
│   │   └── mixed_sampler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── backbone.py
│   │   ├── neck.py
│   │   ├── head.py
│   │   ├── yolox_face_landmark.py
│   │   └── deploy.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── yolox_loss.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── config.py
└── scripts/
    ├── train_phase1.sh
    ├── train_phase2.sh
    └── export_trt.sh

## Datasets

### WIDER FACE

```text
WIDER_ROOT/
├── WIDER_train/
│   └── images/
├── WIDER_val/
│   └── images/
└── wider_face_split/
    ├── wider_face_train_bbx_gt.txt
    └── wider_face_val_bbx_gt.txt
```

### WFLW

```
WFLW_ROOT/
├── WFLW_images/
└── WFLW_annotations/
    └── list_98pt_rect_attr_train_test/
        ├── list_98pt_rect_attr_train.txt
        └── list_98pt_rect_attr_test.txt
```

## Intall
pip install -r requirements.txt
pip install -e .

## Train Phase-1
python tools/train.py --config configs/yolox_l_face_landmark.py --phase phase1

## Train Phase-2
python tools/train.py --config configs/yolox_l_face_landmark.py --phase phase2 --checkpoint outputs/phase1_wider_best.pth

## Export ONNX
python tools/export_onnx.py \
  --config configs/yolox_l_face_landmark.py \
  --checkpoint outputs/final_model.pth \
  --output outputs/model.onnx \
  --dynamic

## Build TensorRT engine
python tools/build_trt.py \
  --onnx outputs/model.onnx \
  --engine outputs/model_fp16.engine \
  --fp16


