"""Normalized Wasserstein Distance (NWD) for tiny-object detection.

Why: YOLO's task-aligned assigner ranks candidate anchors by CIoU. At a 3-14 px
drone, IoU is near-zero and *unstable* (a 1 px shift swings it wildly), so tiny
GTs get almost no positive anchors and their loss barely contributes -- the exact
failure documented in docs/reports/round1-pipeline.md. NWD instead models each box
as a 2-D Gaussian and measures the Wasserstein-2 distance between them, which
degrades *smoothly* with sub-pixel error and does not collapse at small scale
(Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
Detection", AI-TOD AP 11.1 -> 16.1).

`enable_nwd()` monkeypatches two ultralytics internals in place (no fork, no
site-packages edit), guarded to the pinned version's signatures:
  * TaskAlignedAssigner.iou_calculation  -> blend CIoU with NWD (assignment; the
    big lever -- this is what actually gives tiny GTs positive anchors).
  * BboxLoss.forward                     -> blend the (1-IoU) regression term with
    (1-NWD) (smaller lever, keeps gradients alive at few-pixel scale).

Blends (not full replacement) are the default because this is a *generalist*
model spanning 3->100 px across datasets; pure NWD can under-serve the larger
drones. Assigner boxes are in pixels (assign_c ~ object px size); loss boxes are
stride-normalized (loss_c ~ a few cells).
"""
from __future__ import annotations

import torch

_PATCHED = False


def nwd_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor, C: float) -> torch.Tensor:
    """NWD similarity in (0, 1] between paired xyxy boxes (broadcast over leading dims).

    Models a box (cx, cy, w, h) as N([cx, cy], diag((w/2)^2, (h/2)^2)); for
    diagonal Gaussians the squared W2 distance is the plain squared distance of
    the stacked (center, half-extent) vectors. Returns exp(-sqrt(W2^2)/C).
    """
    cx1 = (boxes1[..., 0] + boxes1[..., 2]) * 0.5
    cy1 = (boxes1[..., 1] + boxes1[..., 3]) * 0.5
    w1 = (boxes1[..., 2] - boxes1[..., 0]).abs()
    h1 = (boxes1[..., 3] - boxes1[..., 1]).abs()
    cx2 = (boxes2[..., 0] + boxes2[..., 2]) * 0.5
    cy2 = (boxes2[..., 1] + boxes2[..., 3]) * 0.5
    w2 = (boxes2[..., 2] - boxes2[..., 0]).abs()
    h2 = (boxes2[..., 3] - boxes2[..., 1]).abs()
    w2sq = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
            + ((w1 - w2) * 0.5) ** 2 + ((h1 - h2) * 0.5) ** 2)
    return torch.exp(-torch.sqrt(w2sq + 1e-7) / C)


def enable_nwd(assign_ratio: float = 0.5, assign_c: float = 16.0,
               loss_ratio: float = 0.5, loss_c: float = 2.0) -> None:
    """Install the NWD blends into the ultralytics assigner + bbox loss (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    from ultralytics.utils import tal, loss as ul_loss
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.ops import xywh2xyxy
    from ultralytics.utils.tal import bbox2dist
    import torch.nn.functional as F

    # -- 1. assignment: blend CIoU with NWD when ranking candidate anchors (pixels) --
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        ciou = bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
        nwd = nwd_similarity(gt_bboxes, pd_bboxes, assign_c)
        return (1.0 - assign_ratio) * ciou + assign_ratio * nwd

    tal.TaskAlignedAssigner.iou_calculation = iou_calculation

    # -- 2. regression: blend (1-CIoU) with (1-NWD) (stride-normalized units) --
    def bbox_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                     target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        iou = bbox_iou(pb, tb, xywh=False, CIoU=True)
        nwd = nwd_similarity(pb, tb, loss_c).unsqueeze(-1)
        box_term = (1.0 - iou) * (1.0 - loss_ratio) + (1.0 - nwd) * loss_ratio
        loss_iou = (box_term * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_iou, loss_dfl

    ul_loss.BboxLoss.forward = bbox_forward
    _PATCHED = True
    print(f"[nwd] enabled  assign(ratio={assign_ratio}, C={assign_c})  "
          f"loss(ratio={loss_ratio}, C={loss_c})", flush=True)
