from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from yolox_face.utils.box_ops import bbox_iou, cxcywh_to_xyxy, generalized_iou


class YOLOXLoss(nn.Module):
    def __init__(self, num_classes=1, strides=(8, 16, 32), center_radius=2.5,
                 reg_weight=5.0, obj_weight=1.0, cls_weight=1.0, lmk_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.center_radius = center_radius
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.lmk_weight = lmk_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=1.0 / 9.0)

    @staticmethod
    def _make_grid(hsize, wsize, stride, device):
        yv, xv = torch.meshgrid(torch.arange(hsize, device=device), torch.arange(wsize, device=device), indexing="ij")
        grid = torch.stack((xv, yv), dim=2).view(-1, 2).float()
        strides = torch.full((grid.shape[0], 1), stride, device=device)
        return grid, strides

    def decode_outputs(self, outputs: List[Dict[str, torch.Tensor]]):
        all_cls, all_reg, all_obj, all_lmk, all_grids, all_strides = [], [], [], [], [], []
        for out in outputs:
            cls_pred, reg_pred, obj_pred, lmk_pred, stride = out["cls"], out["reg"], out["obj"], out["lmk"], out["stride"]
            b, _, h, w = cls_pred.shape
            grid, stride_tensor = self._make_grid(h, w, stride, cls_pred.device)
            all_grids.append(grid)
            all_strides.append(stride_tensor)
            all_cls.append(cls_pred.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes))
            all_reg.append(reg_pred.permute(0, 2, 3, 1).reshape(b, -1, 4))
            all_obj.append(obj_pred.permute(0, 2, 3, 1).reshape(b, -1, 1))
            all_lmk.append(lmk_pred.permute(0, 2, 3, 1).reshape(b, -1, 10))

        cls_pred = torch.cat(all_cls, dim=1)
        reg_pred = torch.cat(all_reg, dim=1)
        obj_pred = torch.cat(all_obj, dim=1)
        lmk_pred = torch.cat(all_lmk, dim=1)
        grids = torch.cat(all_grids, dim=0)
        strides = torch.cat(all_strides, dim=0)
        pred_xy = (reg_pred[..., 0:2] + grids[None, ...]) * strides[None, ...]
        pred_wh = reg_pred[..., 2:4].exp() * strides[None, ...]
        pred_boxes = cxcywh_to_xyxy(torch.cat([pred_xy, pred_wh], dim=-1))
        pred_lmk = lmk_pred.view(lmk_pred.shape[0], lmk_pred.shape[1], 5, 2)
        pred_lmk = (pred_lmk + grids[None, :, None, :]) * strides[None, :, None, :]
        return {"cls": cls_pred, "obj": obj_pred, "boxes": pred_boxes, "lmk": pred_lmk, "grids": grids, "strides": strides}

    def _get_in_boxes_info(self, gt_boxes, expanded_strides, xys):
        x_centers = xys[:, 0][None, :]
        y_centers = xys[:, 1][None, :]
        gt_l, gt_t, gt_r, gt_b = gt_boxes[:, 0:1], gt_boxes[:, 1:2], gt_boxes[:, 2:3], gt_boxes[:, 3:4]
        b_l = x_centers - gt_l
        b_r = gt_r - x_centers
        b_t = y_centers - gt_t
        b_b = gt_b - y_centers
        is_in_boxes = torch.stack([b_l, b_r, b_t, b_b], dim=-1).min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        gt_cx = (gt_l + gt_r) * 0.5
        gt_cy = (gt_t + gt_b) * 0.5
        radius = self.center_radius * expanded_strides.squeeze(1)[None, :]
        c_l = x_centers - (gt_cx - radius)
        c_r = (gt_cx + radius) - x_centers
        c_t = y_centers - (gt_cy - radius)
        c_b = (gt_cy + radius) - y_centers
        is_in_centers = torch.stack([c_l, c_r, c_t, c_b], dim=-1).min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def _dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx, pos_idx] = 1

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            multi_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multi_match_mask], dim=0)
            matching_matrix[:, multi_match_mask] = 0
            matching_matrix[cost_argmin, multi_match_mask] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = int(fg_mask_inboxes.sum().item())

        # build a fresh full-size foreground mask instead of modifying fg_mask in-place
        new_fg_mask = torch.zeros_like(fg_mask, dtype=torch.bool)
        new_fg_mask[fg_mask] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, new_fg_mask

    def get_assignments(self, pred_boxes, pred_cls, pred_obj, gt_boxes, gt_classes, strides, grids):
        num_gt = gt_boxes.shape[0]
        num_anchors = pred_boxes.shape[0]
        if num_gt == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
                torch.zeros((0,), device=pred_boxes.device),
                torch.zeros((num_anchors,), dtype=torch.bool, device=pred_boxes.device),
                torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
            )
        xys = (grids + 0.5) * strides
        fg_mask, is_in_boxes_and_center = self._get_in_boxes_info(gt_boxes, strides, xys)
        valid_pred_boxes = pred_boxes[fg_mask]
        valid_pred_cls = pred_cls[fg_mask]
        valid_pred_obj = pred_obj[fg_mask]
        pair_wise_ious = bbox_iou(gt_boxes, valid_pred_boxes)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, valid_pred_boxes.shape[0], 1)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        with autocast("cuda", enabled=False):
            cls_preds_ = (valid_pred_cls.float().sigmoid_() * valid_pred_obj.float().sigmoid_()).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1), gt_cls_per_image, reduction="none").sum(-1)
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center)
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask = self._dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask
        )
        return gt_matched_classes, pred_ious_this_matching, fg_mask, matched_gt_inds

    def forward(self, outputs, batch):
        decoded = self.decode_outputs(outputs)
        pred_boxes, pred_cls, pred_obj, pred_lmk = decoded["boxes"], decoded["cls"], decoded["obj"], decoded["lmk"]
        grids, strides = decoded["grids"], decoded["strides"]
        batch_boxes, batch_labels, batch_landmarks = batch["boxes"], batch["labels"], batch["landmarks"]
        has_det, has_lmk = batch["has_det"], batch["has_lmk"]
        device = pred_boxes.device
        total_num_fg = 0.0
        loss_iou = pred_boxes.sum() * 0.0
        loss_obj = pred_boxes.sum() * 0.0
        loss_cls = pred_boxes.sum() * 0.0
        loss_lmk = pred_boxes.sum() * 0.0

        for b in range(pred_boxes.shape[0]):
            gt_boxes = batch_boxes[b].to(device)
            gt_classes = batch_labels[b].to(device)
            gt_landmarks = batch_landmarks[b].to(device)
            obj_target = torch.zeros((pred_boxes.shape[1], 1), device=device)
            fg_mask = torch.zeros((pred_boxes.shape[1],), dtype=torch.bool, device=device)

            if has_det[b] and gt_boxes.numel() > 0:
                matched_classes, pred_ious_this_matching, fg_mask, matched_gt_inds = self.get_assignments(
                    pred_boxes[b], pred_cls[b], pred_obj[b], gt_boxes, gt_classes, strides, grids
                )
                num_fg_img = fg_mask.sum().item()
                total_num_fg += num_fg_img
                if num_fg_img > 0:
                    cls_target = F.one_hot(matched_classes.to(torch.int64), self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                    reg_target = gt_boxes[matched_gt_inds]
                    obj_target[fg_mask] = 1.0
                    loss_iou = loss_iou + (1.0 - generalized_iou(pred_boxes[b][fg_mask], reg_target)).sum()
                    loss_cls = loss_cls + self.bce(pred_cls[b][fg_mask], cls_target).sum()
                    if has_lmk[b] and gt_landmarks.numel() > 0:
                        lmk_target = gt_landmarks[matched_gt_inds]
                        valid = (lmk_target[..., 0] >= 0) & (lmk_target[..., 1] >= 0)
                        if valid.any():
                            lmk_loss_raw = self.smooth_l1(pred_lmk[b][fg_mask], lmk_target)
                            lmk_mask = valid.unsqueeze(-1).float()
                            loss_lmk = loss_lmk + (lmk_loss_raw * lmk_mask).sum() / lmk_mask.sum().clamp(min=1.0)
            loss_obj = loss_obj + self.bce(pred_obj[b], obj_target).sum()

        num_fg = max(total_num_fg, 1.0)
        loss = self.reg_weight * loss_iou / num_fg + self.obj_weight * loss_obj / num_fg + self.cls_weight * loss_cls / num_fg + self.lmk_weight * loss_lmk / num_fg
        return {
            "loss": loss,
            "loss_iou": self.reg_weight * loss_iou / num_fg,
            "loss_obj": self.obj_weight * loss_obj / num_fg,
            "loss_cls": self.cls_weight * loss_cls / num_fg,
            "loss_lmk": self.lmk_weight * loss_lmk / num_fg,
        }