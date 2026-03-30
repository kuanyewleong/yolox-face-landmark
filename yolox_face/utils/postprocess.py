import torch
import torchvision


def postprocess_single(pred_boxes, pred_obj, pred_cls, pred_lmk, conf_thres=0.25, nms_thres=0.5):
    scores = pred_obj.sigmoid().squeeze(-1) * pred_cls.sigmoid().max(dim=-1).values
    keep = scores > conf_thres
    if keep.sum() == 0:
        return {
            "boxes": pred_boxes.new_zeros((0, 4)),
            "scores": pred_boxes.new_zeros((0,)),
            "landmarks": pred_boxes.new_zeros((0, 5, 2)),
        }
    boxes = pred_boxes[keep]
    scores = scores[keep]
    landmarks = pred_lmk[keep]
    keep_idx = torchvision.ops.nms(boxes, scores, nms_thres)
    return {
        "boxes": boxes[keep_idx],
        "scores": scores[keep_idx],
        "landmarks": landmarks[keep_idx],
    }