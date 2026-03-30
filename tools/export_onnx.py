import argparse
import torch
from yolox_face.config import load_config
from yolox_face.models.deploy import DeploymentWrapper
from yolox_face.models.yolox_face_landmark import YOLOXFaceLandmark
from yolox_face.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = YOLOXFaceLandmark(**cfg["model"])
    load_checkpoint(model, args.checkpoint)
    model.eval()
    wrapper = DeploymentWrapper(model)
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    dynamic_shapes = None
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "batch", 1: "anchors"},
            "obj": {0: "batch", 1: "anchors"},
            "cls": {0: "batch", 1: "anchors"},
            "lmk": {0: "batch", 1: "anchors"},
        }
        dynamic_shapes = {"images": {0: torch.export.Dim("batch"), 2: torch.export.Dim("height"), 3: torch.export.Dim("width")}}

    torch.onnx.export(
        wrapper,
        (dummy,),
        args.output,
        input_names=["images"],
        output_names=["boxes", "obj", "cls", "lmk"],
        opset_version=args.opset,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()