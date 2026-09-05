"""
Improved Loss Functions for YOLO v8
Includes: Focal Loss, GIoU/DIoU/CIoU Loss, and Alpha-IoU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss from "Focal Loss for Dense Object Detection"
    
    Addresses class imbalance by down-weighting easy examples
    and focusing on hard negatives.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor in range (0, 1) to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t) 
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (batch, num_classes) or (batch, num_anchors, num_classes)
            target: Target labels (batch,) or (batch, num_anchors)
            
        Returns:
            Focal loss value
        """
        # Convert to probabilities
        pred_sigmoid = torch.sigmoid(pred)
        
        # Create one-hot encoding if needed
        if target.dim() == 1 or target.shape != pred.shape:
            target_one_hot = torch.zeros_like(pred)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        else:
            target_one_hot = target
            
        # Calculate p_t
        p_t = pred_sigmoid * target_one_hot + (1 - pred_sigmoid) * (1 - target_one_hot)
        
        # Calculate alpha_t
        alpha_t = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
        
        # Calculate focal weight
        focal_weight = torch.pow(1 - p_t, self.gamma)
        
        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot.float(), reduction='none'
        )
        
        # Combine
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss from "Generalized Focal Loss"
    
    Combines classification score with localization quality
    for more accurate detection.
    """
    
    def __init__(self, beta=2.0, use_sigmoid=True):
        """
        Args:
            beta: Modulating factor
            use_sigmoid: Use sigmoid activation
        """
        super().__init__()
        self.beta = beta
        self.use_sigmoid = use_sigmoid
        
    def forward(self, pred, target, quality_score=None):
        """
        Args:
            pred: Predictions
            target: Target labels  
            quality_score: IoU or other quality metric
            
        Returns:
            Quality focal loss
        """
        # Scale to range [0, 1]
        pred_sigmoid = torch.sigmoid(pred)
        
        if quality_score is None:
            quality_score = torch.ones_like(target)
            
        # Scale target by quality score
        target_scaled = target * quality_score
        
        # Focal term
        focal_term = (pred_sigmoid - target_scaled).abs().pow(self.beta)
        
        # Loss
        loss = F.binary_cross_entropy_with_logits(
            pred, target_scaled, reduction='none'
        ) * focal_term
        
        return loss.mean()


class GIoULoss(nn.Module):
    """
    GIoU Loss from "Generalized Intersection over Union"
    
    Better than IoU loss as it accounts for enclosing boxes.
    GIoU = IoU - |C - A U B| / |C|
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted boxes in xyxy format (N, 4)
            target: Target boxes in xyxy format (N, 4)
            
        Returns:
            1 - GIoU loss
        """
        # Calculate areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Enclosing box
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DIoULoss(nn.Module):
    """
    DIoU Loss from "Distance-IoU Loss: Faster and Better Learning for 
    Bounding Box Regression"
    
    DIoU = IoU - R_DIoU
    where R_DIoU penalizes center distance
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted boxes in xyxy format (N, 4)
            target: Target boxes in xyxy format (N, 4)
            
        Returns:
            1 - DIoU loss
        """
        # Calculate areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Centers
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        
        # Center distance squared
        center_dist_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
        
        # Enclosing box diagonal squared
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        diagonal_sq = (enclose_x2 - enclose_x1).pow(2) + \
                      (enclose_y2 - enclose_y1).pow(2)
        
        # DIoU
        diou = iou - center_dist_sq / (diagonal_sq + 1e-7)
        
        loss = 1 - diou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CIoULoss(nn.Module):
    """
    CIoU Loss from "Distance-IoU Loss"
    
    CIoU = IoU - (center_dist^2 / diagonal^2) - alpha * v
    where v measures aspect ratio consistency
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted boxes in xyxy format (N, 4)
            target: Target boxes in xyxy format (N, 4)
            
        Returns:
            1 - CIoU loss
        """
        # Calculate areas and IoU
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-7)
        
        # Center distance
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        
        center_dist_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
        
        # Enclosing diagonal
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        diagonal_sq = (enclose_x2 - enclose_x1).pow(2) + \
                      (enclose_y2 - enclose_y1).pow(2)
        
        # Aspect ratio consistency
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
        
        v = (4 / math.pi**2) * torch.pow(
            torch.atan(target_w / (target_h + 1e-7)) - 
            torch.atan(pred_w / (pred_h + 1e-7)), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU
        ciou = iou - center_dist_sq / (diagonal_sq + 1e-7) - alpha * v
        
        loss = 1 - ciou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AlphaIoULoss(nn.Module):
    """
    Alpha-IoU Loss from "Alpha-IoU: A New Family of Power'd IoU Losses
    for Bounding Box Regression"
    
    Gives more weight to hard examples through power transformation.
    Alpha-IoU = IoU^alpha - (1-IoU)^alpha
    """
    
    def __init__(self, alpha=3.0, iou_type='ciou', reduction='mean'):
        """
        Args:
            alpha: Power parameter (higher = more focus on hard examples)
            iou_type: 'iou' | 'giou' | 'diou' | 'ciou'
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.iou_type = iou_type
        self.reduction = reduction
        
        # Initialize base IoU loss
        if iou_type == 'giou':
            self.iou_loss = GIoULoss(reduction='none')
        elif iou_type == 'diou':
            self.iou_loss = DIoULoss(reduction='none')
        elif iou_type == 'ciou':
            self.iou_loss = CIoULoss(reduction='none')
        else:
            self.iou_loss = None  # Plain IoU
            
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted boxes (N, 4)
            target: Target boxes (N, 4)
            
        Returns:
            Alpha-IoU loss
        """
        if self.iou_loss is not None:
            # Use specific IoU type
            iou_values = 1 - self.iou_loss(pred, target)
        else:
            # Plain IoU
            inter_x1 = torch.max(pred[:, 0], target[:, 0])
            inter_y1 = torch.max(pred[:, 1], target[:, 1])
            inter_x2 = torch.min(pred[:, 2], target[:, 2])
            inter_y2 = torch.min(pred[:, 3], target[:, 3])
            
            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                         torch.clamp(inter_y2 - inter_y1, min=0)
            
            pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
            target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
            union_area = pred_area + target_area - inter_area
            
            iou_values = inter_area / (union_area + 1e-7)
        
        # Alpha-IoU transformation
        # Note: This is a simplified version. Full paper has more details.
        alpha_iou = torch.pow(iou_values, self.alpha)
        
        loss = 1 - alpha_iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SIoULoss(nn.Module):
    """
    SIoU Loss from "SIoU Loss: More Powerful Learning for Bounding Box Regression"
    
    Considers angle of vector between box centers for better convergence.
    """
    
    def __init__(self, theta=4.0, reduction='mean'):
        """
        Args:
            theta: Angle cost parameter
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.theta = theta
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted boxes in xyxy format (N, 4)
            target: Target boxes in xyxy format (N, 4)
            
        Returns:
            SIoU loss
        """
        # Calculate IoU
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-7)
        
        # Centers
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        
        # Distance
        dx = pred_cx - target_cx
        dy = pred_cy - target_cy
        sigma = torch.sqrt(dx.pow(2) + dy.pow(2))
        
        sin_alpha = torch.abs(dx) / (sigma + 1e-7)
        sin_beta = torch.abs(dy) / (sigma + 1e-7)
        sin_alpha = torch.clamp(sin_alpha, 0, 1)
        sin_beta = torch.clamp(sin_beta, 0, 1)
        
        # Angle cost
        threshold = pow(2, 0.5) / 2
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi/2)
        
        # Distance cost
        gamma = angle_cost - 2
        omega = (1 - gamma * sin_alpha.pow(self.theta)) * sin_alpha - sin_beta
        distance_cost = 2 - omega
        
        # Enclosing box
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        
        # Width/height cost
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
        
        eps = 1e-7
        cw = torch.max(pred_w, target_w)
        ch = torch.max(pred_h, target_h)
        
        w_cost = torch.pow(pred_w - target_w, 2) / (cw.pow(2) + eps)
        h_cost = torch.pow(pred_h - target_h, 2) / (ch.pow(2) + eps)
        
        shape_cost = w_cost + h_cost
        
        # SIoU
        siou = iou - (distance_cost + shape_cost) / 2
        
        loss = 1 - siou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Helper function to create loss function based on name
def create_bbox_loss(loss_type='ciou', **kwargs):
    """
    Create bounding box loss function
    
    Args:
        loss_type: 'iou' | 'giou' | 'diou' | 'ciou' | 'alphaiou' | 'siou'
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
    """
    loss_map = {
        'iou': None,  # Plain IoU
        'giou': GIoULoss,
        'diou': DIoULoss,
        'ciou': CIoULoss,
        'alphaiou': AlphaIoULoss,
        'siou': SIoULoss,
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Available: {list(loss_map.keys())}")
    
    loss_fn = loss_map[loss_type]
    
    if loss_fn is None:
        # Return simple IoU loss
        return lambda pred, target: 1 - calculate_iou(pred, target)
    
    return loss_fn(**kwargs)


def calculate_iou(pred, target, eps=1e-7):
    """Calculate IoU between predicted and target boxes"""
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union_area = pred_area + target_area - inter_area
    
    return inter_area / (union_area + eps)
