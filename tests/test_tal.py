# Ultralytics YOLO 🚀, GPL-3.0 license
"""TaskAlignedAssigner memory-saving rewrites must give exactly the same assignment as the original formulation."""
import torch
import torch.nn.functional as F

from ultralytics.yolo.utils import tal
from ultralytics.yolo.utils.metrics import bbox_iou


class _ReferenceAssigner(tal.TaskAlignedAssigner):
    """Original (8.0.8) one_hot / full-tensor implementations."""

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.long().squeeze(-1)
        bbox_scores = pd_scores[ind[0], :, ind[1]]
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
        return bbox_scores.pow(self.alpha) * overlaps.pow(self.beta), overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        is_in_topk = F.one_hot(topk_idxs, metrics.shape[-1]).sum(-2)
        return torch.where(is_in_topk > 1, 0, is_in_topk).to(metrics.dtype)


def _inputs(b=3, n=40, imgsz=320, seed=0):
    torch.manual_seed(seed)
    feats = [torch.zeros(1, 1, imgsz // s, imgsz // s) for s in (8, 16, 32)]
    anc, stride = tal.make_anchors(feats, [8, 16, 32])
    anc = anc * stride
    gt, mask = torch.zeros(b, n, 4), torch.zeros(b, n, 1)
    for i in range(b):
        k = n - 7 * i
        c, wh = torch.rand(k, 2) * imgsz, 10 + torch.rand(k, 2) * 40
        gt[i, :k], mask[i, :k] = torch.cat([c - wh / 2, c + wh / 2], 1).clamp(0, imgsz), 1
    a = anc.shape[0]
    ctr, wh = anc[None].expand(b, -1, -1) + torch.randn(b, a, 2) * 4, 10 + torch.rand(b, a, 2) * 40
    return torch.rand(b, a, 2), torch.cat([ctr - wh / 2, ctr + wh / 2], -1), anc, torch.randint(0, 2, (b, n, 1)).float(), gt, mask


def test_assigner_matches_reference():
    for seed in range(3):
        args = _inputs(seed=seed)
        new = tal.TaskAlignedAssigner(topk=10, num_classes=2, alpha=0.5, beta=6.0)
        new.chunk_elems = 5000  # force several CIoU chunks
        ref = _ReferenceAssigner(topk=10, num_classes=2, alpha=0.5, beta=6.0)
        for x, y in zip(new(*args), ref(*args)):
            assert torch.equal(x, y)


def test_select_candidates_in_gts_matches_reference():
    _, _, anc, _, gt, _ = _inputs()
    lt, rb = gt.view(-1, 1, 4).chunk(2, 2)
    ref = torch.cat((anc[None] - lt, rb - anc[None]), dim=2).view(*gt.shape[:2], anc.shape[0], -1).amin(3).gt_(1e-9)
    assert torch.equal(tal.select_candidates_in_gts(anc, gt), ref)


def test_assigner_matches_reference_with_half_scores():
    # under AMP the trainer passes fp16 scores and fp32 boxes; overlaps must stay fp32 like the original
    pd_scores, pd_bboxes, anc, labels, gt, mask = _inputs(seed=5)
    args = (pd_scores.half(), pd_bboxes, anc, labels, gt, mask)
    new = tal.TaskAlignedAssigner(topk=10, num_classes=2, alpha=0.5, beta=6.0)
    ref = _ReferenceAssigner(topk=10, num_classes=2, alpha=0.5, beta=6.0)
    for x, y in zip(new(*args), ref(*args)):
        assert x.dtype == y.dtype and torch.equal(x, y)
