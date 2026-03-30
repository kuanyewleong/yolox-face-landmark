import argparse
import os
from torch.utils.data import ConcatDataset, DataLoader
from yolox_face.config import load_config
from yolox_face.data.collate import collate_face
from yolox_face.data.mixed_sampler import RatioConcatSampler
from yolox_face.data.transforms import FaceTrainTransform, FaceValTransform
from yolox_face.data.wflw import WFLWDataset
from yolox_face.data.wider_face import WiderFaceDataset
from yolox_face.engine.trainer import run_phase
from yolox_face.models.yolox_face_landmark import YOLOXFaceLandmark
from yolox_face.utils.checkpoint import load_checkpoint, save_final_model
from yolox_face.utils.env import set_seed


def build_model(cfg):
    return YOLOXFaceLandmark(**cfg["model"])


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


def build_loaders(cfg, phase):
    train_size = (cfg["train"]["input_size"], cfg["train"]["input_size"])
    train_tf = FaceTrainTransform(train_size)
    val_tf = FaceValTransform(train_size)

    wider_train = WiderFaceDataset(cfg["data"]["wider_root"], split="train", transform=train_tf)
    wider_val = WiderFaceDataset(cfg["data"]["wider_root"], split="val", transform=val_tf)
    wflw_train = WFLWDataset(cfg["data"]["wflw_root"], split="train", transform=train_tf)
    wflw_val = WFLWDataset(cfg["data"]["wflw_root"], split="test", transform=val_tf)

    if phase == "phase1":
        train_loader = make_loader(wider_train, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=True)
        val_loader = make_loader(wider_val, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=False)
        return train_loader, val_loader

    if phase == "phase2":
        mixed_train = ConcatDataset([wider_train, wflw_train])
        epoch_samples = len(wider_train)
        sampler = RatioConcatSampler(
            mixed_train,
            num_samples=epoch_samples,
            wider_ratio=cfg["train"]["mixed_wider_ratio"],
            seed=cfg["train"]["seed"],
        )
        train_loader = make_loader(mixed_train, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=False, sampler=sampler)
        val_loader = make_loader(wflw_val, cfg["train"]["batch_size"], cfg["train"]["workers"], shuffle=False)
        return train_loader, val_loader

    raise ValueError(f"Unsupported phase: {phase}")


def save_phase_artifact(model, cfg, filename):
    path = os.path.join(cfg["output_dir"], filename)
    save_final_model(model, cfg, path)
    print(f"Saved model to {path}")


def train_phase(model, cfg, phase, checkpoint_path=None, resume=False):
    train_loader, val_loader = build_loaders(cfg, phase)
    output_prefix = "phase1_wider" if phase == "phase1" else "phase2_multitask"
    trained_model = run_phase(
        model,
        train_loader,
        val_loader,
        cfg,
        output_prefix=output_prefix,
        phase_name=phase,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    return trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default="all", choices=["phase1", "phase2", "all"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    model = build_model(cfg)

    if args.phase == "phase1":
        model = train_phase(
            model,
            cfg,
            phase="phase1",
            checkpoint_path=args.checkpoint or None,
            resume=args.resume,
        )
        save_phase_artifact(model, cfg, "phase1_wider_final.pth")
        return

    if args.phase == "phase2":
        if not args.checkpoint:
            raise ValueError("Phase 2 requires --checkpoint pointing to a phase 1 or compatible pretrained checkpoint.")
        load_checkpoint(model, args.checkpoint, strict=False)
        print(f"Loaded phase 2 initialization weights from {args.checkpoint}")
        model = train_phase(
            model,
            cfg,
            phase="phase2",
            checkpoint_path=args.checkpoint if args.resume else None,
            resume=args.resume,
        )
        save_phase_artifact(model, cfg, "phase2_multitask_final.pth")
        save_phase_artifact(model, cfg, "final_model.pth")
        return

    # all
    model = train_phase(
        model,
        cfg,
        phase="phase1",
        checkpoint_path=args.checkpoint or None,
        resume=args.resume,
    )
    phase1_final = os.path.join(cfg["output_dir"], "phase1_wider_final.pth")
    save_final_model(model, cfg, phase1_final)
    print(f"Saved model to {phase1_final}")

    model = build_model(cfg)
    load_checkpoint(model, phase1_final, strict=False)
    print(f"Loaded phase 2 initialization weights from {phase1_final}")
    model = train_phase(model, cfg, phase="phase2", checkpoint_path=None, resume=False)
    save_phase_artifact(model, cfg, "phase2_multitask_final.pth")
    save_phase_artifact(model, cfg, "final_model.pth")


if __name__ == "__main__":
    main()