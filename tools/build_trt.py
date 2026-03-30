import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--min-shape", default="1x3x320x320")
    parser.add_argument("--opt-shape", default="1x3x640x640")
    parser.add_argument("--max-shape", default="4x3x960x960")
    args = parser.parse_args()

    cmd = [
        "trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--memPoolSize=workspace:{args.workspace_gb * 1024}",
        "--builderOptimizationLevel=5",
        "--useCudaGraph",
        f"--minShapes=images:{args.min_shape}",
        f"--optShapes=images:{args.opt_shape}",
        f"--maxShapes=images:{args.max_shape}",
    ]
    if args.fp16:
        cmd.append("--fp16")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()