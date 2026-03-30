import argparse
import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image_path, input_size=640):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    scale = min(input_size / w0, input_size / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    dx = (input_size - nw) // 2
    dy = (input_size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    x = canvas.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return x, img, scale, dx, dy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    x, img, scale, dx, dy = preprocess(args.image, args.input_size)
    boxes, obj, cls, lmk = sess.run(None, {"images": x})
    print("boxes:", boxes.shape, "obj:", obj.shape, "cls:", cls.shape, "lmk:", lmk.shape)


if __name__ == "__main__":
    main()