import argparse
from torch.utils.data import ConcatDataset, DataLoader
from yolox_face.config import load_config
from yolox_face.data.collate import collate_face
from yolox_face.data.mixed_sampler import RatioConcatSampler
from yolox_face.data.transforms import FaceTrainTransform, FaceValTransform
from yolox_face.data.wflw import WFLWDataset
from yolox_face.data.wider_face import WiderFaceDataset
from yolox_face.engine.trainer import run_phase
from yolox_face.models.yolox_face_landmark import YOLOXFaceLandmark
from yolox_face.utils.checkpoint import load_checkpoint
from yolox_face.utils.env import set_seed


def make_loader(dataset, batch_size, workers, shuffle=True, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=collate_face,
        persistent_workers=workers > 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default="all", choices=["phase1", "phase2", "all"])
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    train_tf = FaceTrainTransform((cfg["train"]["input_size"], cfg["train"]["input_size"]))
    val_tf = FaceValTransform((cfg["train"]["input_size"], cfg["train"]["input_size"]))

    wider_train = WiderFaceDataset(cfg["data"]["wider_root"], split="train", transform=train_tf)
    wider_val = WiderFaceDataset(cfg["data"]["wider_root"], split="val", transform=val_tf)
    wflw_train = WFLWDataset(cfg["data"]["wflw_root"], split="train", transform=train_tf)
    wflw_val = WFLWDataset(cfg["data"]["wflw_root"], split="test", transform=val_tf)

    model = YOLOXFaceLandmark(**cfg["model"])
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    if args.phase in ["phase1", "all"]:
        train_loader = make_loader(wider_train, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=True)
        val_loader = make_loader(wider_val, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=False)
        model = run_phase(model, train_loader, val_loader, cfg, "phase1_wider", "phase1")

    if args.phase in ["phase2", "all"]:
        mixed_train = ConcatDataset([wider_train, wflw_train])
        epoch_samples = len(wider_train)
        sampler = RatioConcatSampler(mixed_train, num_samples=epoch_samples, wider_ratio=cfg["train"]["mixed_wider_ratio"], seed=cfg["train"]["seed"])
        train_loader = make_loader(mixed_train, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=False, sampler=sampler)
        val_loader = make_loader(wflw_val, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=False)
        model = run_phase(model, train_loader, val_loader, cfg, "phase2_multitask", "phase2")


if __name__ == "__main__":
    main()