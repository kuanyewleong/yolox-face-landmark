import argparse
import torch
from torch.utils.data import DataLoader
from yolox_face.config import load_config
from yolox_face.data.collate import collate_face
from yolox_face.data.transforms import FaceValTransform
from yolox_face.data.wflw import WFLWDataset
from yolox_face.data.wider_face import WiderFaceDataset
from yolox_face.engine.evaluator import evaluate_simple
from yolox_face.losses.yolox_loss import YOLOXLoss
from yolox_face.models.yolox_face_landmark import YOLOXFaceLandmark
from yolox_face.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["wider", "wflw"], default="wider")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = YOLOXFaceLandmark(**cfg["model"])
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tf = FaceValTransform((cfg["train"]["input_size"], cfg["train"]["input_size"]))
    if args.dataset == "wider":
        ds = WiderFaceDataset(cfg["data"]["wider_root"], split="val", transform=tf)
    else:
        ds = WFLWDataset(cfg["data"]["wflw_root"], split="test", transform=tf)

    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["workers"], collate_fn=collate_face)
    loss_fn = YOLOXLoss(num_classes=cfg["model"]["num_classes"])
    print(evaluate_simple(model, loader, device, loss_fn))


if __name__ == "__main__":
    main()