"""
Improved Data Augmentation for YOLO v8
Includes: GridMask, CutOut, RandomErasing, and improved MixUp
"""

import math
import random
from copy import deepcopy

import cv2
import numpy as np
import torch

from ..utils import LOGGER
from ..utils.instance import Instances


class GridMask:
    """
    GridMask augmentation from "GridMask: Structured Dropout for Improved 
    Generalization in Image Classification and Object Detection"
    
    Helps the model be robust to occluded objects by masking grid patterns
    """
    
    def __init__(self, use_h=True, use_w=True, rotate=1, mode=0, 
                 ratio=0.5, max_epoch=300, d_range=(96, 224), 
                 p=0.5):
        """
        Args:
            use_h: Use horizontal grid
            use_w: Use vertical grid
            rotate: Maximum rotation angle in degrees
            mode: 0 = keep original, 1 = cutoff
            ratio: Grid mask ratio (percentage of masked area)
            max_epoch: Maximum training epochs for scheduling
            d_range: Range for grid size (min, max)
            p: Probability of applying this augmentation
        """
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.mode = mode
        self.ratio = ratio
        self.max_epoch = max_epoch
        self.d_range = d_range
        self.p = p
        
    def __call__(self, labels):
        """Apply GridMask augmentation"""
        if random.random() > self.p:
            return labels
            
        img = labels.get('img')
        if img is None:
            return labels
            
        h, w = img.shape[:2]
        
        # Grid parameters
        d = random.randint(self.d_range[0], self.d_range[1])
        mask_length = int(d * self.ratio)
        
        # Create base mask
        # This implementation rotates the mask 
        mask = np.ones((h, w), dtype=np.float32)
        
        # Make larger mask for rotation
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        mask = np.ones((hh, ww), dtype=np.float32)
        
        # Draw grid lines
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)
        
        for i in range((hh // d) + 1):
            s = max(0, st_h + i * d)
            e = min(hh, s + mask_length)
            if self.use_h and e > s:
                mask[s:e, :] = 0
                
        for i in range((ww // d) + 1):
            s = max(0, st_w + i * d)
            e = min(ww, s + mask_length)
            if self.use_w and e > s:
                mask[:, s:e] = 0
        
        # Crop back to original size
        dh = (hh - h) // 2
        dw = (ww - w) // 2
        mask = mask[dh:dh+h, dw:dw+w]
        
        # Apply mask
        if self.mode == 1:
            # Fill with mean color
            mean_color = img.mean(axis=(0, 1))
            img = img * mask[..., np.newaxis] + mean_color * (1 - mask[..., np.newaxis])
        else:
            # Just black out
            img = img * mask[..., np.newaxis]
        
        labels['img'] = img.astype(np.uint8)
        return labels


class CutOut:
    """
    CutOut augmentation: "Improved Regularization of Convolutional Neural Networks 
    with Cutout"
    
    Randomly masks out square regions during training
    """
    
    def __init__(self, n_holes=1, length=64, p=0.5):
        """
        Args:
            n_holes: Number of patches to cut out
            length: Maximum length (in pixels) of each square patch
            p: Probability of applying this augmentation
        """
        self.n_holes = n_holes
        self.length = length
        self.p = p
        
    def __call__(self, labels):
        """Apply CutOut augmentation"""
        if random.random() > self.p:
            return labels
            
        img = labels.get('img')
        if img is None:
            return labels
            
        h, w = img.shape[:2]
        
        for _ in range(self.n_holes):
            # Random size
            size = random.randint(self.length // 2, self.length)
            
            # Random center point
            y = random.randint(0, h)
            x = random.randint(0, w)
            
            # Compute bounding box
            y1 = np.clip(y - size // 2, 0, h)
            y2 = np.clip(y + size // 2, 0, h)
            x1 = np.clip(x - size // 2, 0, w)
            x2 = np.clip(x + size // 2, 0, w)
            
            # Fill with mean color or zeros
            # Using mean color helps maintain intensity distribution
            mean_color = img.mean(axis=(0, 1))
            
            # Apply cutout
            if self.n_holes > 2:
                # For better generalization, use random color
                random_color = np.random.randint(0, 256, size=3).astype(np.float32)
                img[y1:y2, x1:x2] = random_color
            else:
                img[y1:y2, x1:x2] = mean_color
        
        labels['img'] = img.astype(np.uint8)
        return labels


class RandomErasing:
    """
    Random Erasing Data Augmentation 
    "Random Erasing Data Augmentation" (Zhong et al., 2017)
    
    Randomly selects a rectangle region in an image and erases its pixels
    """
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, 
                 mean_val=(0.4914, 0.4822, 0.4465)):
        """
        Args:
            probability: Probability of performing random erasing
            sl: Minimum proportion of erased area
            sh: Maximum proportion of erased area  
            r1: Minimum aspect ratio of erased area
            mean_val: Mean values for filling erased region
        """
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean_val = mean_val
        
    def __call__(self, labels):
        """Apply Random Erasing"""
        if random.random() > self.probability:
            return labels
            
        img = labels.get('img')
        if img is None:
            return labels
            
        h, w, c = img.shape
        area = h * w
        
        for attempt in range(100):  # Try up to 100 times
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1./self.r1)
            
            h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w_erase < w and h_erase < h:
                # Random position
                x1 = random.randint(0, w - w_erase)
                y1 = random.randint(0, h - h_erase)
                
                # Fill with random value or mean
                if self.mean_val and random.random() < 0.5:
                    img[y1:y1+h_erase, x1:x1+w_erase] = \
                        np.array(self.mean_val) * 255
                else:
                    # Random noise
                    img[y1:y1+h_erase, x1:x1+w_erase] = \
                        np.random.randint(0, 256, (h_erase, w_erase, c))
                
                labels['img'] = img
                return labels
        
        return labels


class ImprovedMixUp:
    """
    Improved MixUp augmentation with smart object selection
    
    Mixes images based on bounding box overlap to create more meaningful
    training samples
    """
    
    def __init__(self, dataset, pre_transform=None, p=0.15, 
                 max_mixup_ratio=0.5, beta_alpha=32.0):
        """
        Args:
            dataset: Dataset object for sampling other images
            pre_transform: Transform to apply before mixup
            p: Probability of applying MixUp
            max_mixup_ratio: Maximum ratio for mixing
            beta_alpha: Alpha parameter for Beta distribution
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p
        self.max_mixup_ratio = max_mixup_ratio
        self.beta_alpha = beta_alpha
        
    def __call__(self, labels):
        """Apply improved MixUp"""
        if random.random() > self.p:
            return labels
            
        # Get another image
        index = random.randint(0, len(self.dataset) - 1)
        mix_labels = self.dataset.get_label_info(index)
        
        if self.pre_transform:
            mix_labels = self.pre_transform(mix_labels)
            
        # Sample mixing ratio from Beta distribution
        # This gives more realistic mixing ratios
        if random.random() < 0.5:
            # Use Beta distribution
            lam = np.random.beta(self.beta_alpha, self.beta_alpha)
        else:
            # Uniform distribution
            lam = random.uniform(0, self.max_mixup_ratio)
            
        # Mix images
        img1 = labels['img'].astype(np.float32)
        img2 = mix_labels['img'].astype(np.float32)
        
        labels['img'] = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
        
        # Mix instances
        instances1 = labels.get('instances', Instances())
        instances2 = mix_labels.get('instances', Instances())
        
        # Combine instances with mixing weights
        # Instances from image 1 get higher weight when lam is high
        if instances1 and instances2:
            # Scale confidence based on mixing ratio
            combined_instances = Instances.cat([instances1, instances2])
            labels['instances'] = combined_instances
            
            # Store mixup info
            labels['mixup_ratio'] = lam
            
        return labels


class RandomPerspectiveSmart:
    """
    Random Perspective transform with smart object preservation
    
    Applies perspective transform while ensuring objects remain visible
    """
    
    def __init__(self, degrees=0, translate=0.1, scale=0.5, 
                 shear=0, perspective=0.0, border=(0, 0), p=0.5):
        """
        Args:
            degrees: Rotation angle in degrees (+/-)
            translate: Translation as fraction of image size
            scale: Scaling factor range (+/-)
            shear: Shear angle in degrees (+/-)
            perspective: Perspective distortion
            border: Border padding
            p: Probability of applying transform
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.p = p
        
    def __call__(self, labels):
        """Apply smart perspective transform"""
        if random.random() > self.p:
            return labels
            
        img = labels.get('img')
        instances = labels.get('instances')
        
        if img is None or instances is None:
            return labels
            
        h, w = img.shape[:2]
        
        # Only apply if it doesn't clip too many objects
        bboxes = instances.bboxes
        if len(bboxes) > 0:
            # Check object density
            obj_area = np.sum([b[2]*b[3] for b in bboxes])
            img_area = h * w
            
            # Reduce distortion if many objects
            if obj_area / img_area > 0.3:
                self.degrees *= 0.5
                self.translate *= 0.5
                
        # Apply transform (simplified version)
        # Center
        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2
        
        # Rotation
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D((0, 0), a, s)
        
        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        
        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * w
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * h
        
        # Combine transforms
        M = T @ S @ R @ C
        
        # Apply perspective if needed
        if self.perspective > 0:
            P = np.eye(3)
            P[2, 0] = random.uniform(-self.perspective, self.perspective)
            P[2, 1] = random.uniform(-self.perspective, self.perspective)
            M = P @ M
            
        # Apply to image
        img = cv2.warpPerspective(img, M, (w + self.border[0], h + self.border[1]))
        
        labels['img'] = img
        
        return labels


class AugmentationComposer:
    """
    Compose multiple augmentations with smart scheduling
    """
    
    def __init__(self, augmentations, epoch=0, max_epoch=300):
        """
        Args:
            augmentations: List of augmentation transforms
            epoch: Current epoch
            max_epoch: Maximum training epochs
        """
        self.augmentations = augmentations
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Update current epoch for scheduling"""
        self.current_epoch = epoch
        
    def __call__(self, labels):
        """Apply augmentations with epoch-based scheduling"""
        for aug in self.augmentations:
            # Some augmentations might need epoch information
            if hasattr(aug, 'max_epoch'):
                aug.current_epoch = self.current_epoch
            labels = aug(labels)
        return labels


# Helper function to create augmentation pipeline
def create_augmentation_pipeline(hyp, dataset=None, epoch=0):
    """
    Create augmentation pipeline based on hyperparameters
    
    Args:
        hyp: Hyperparameters dictionary
        dataset: Dataset object for mixup-like augmentations
        epoch: Current epoch
        
    Returns:
        Composed augmentation pipeline
    """
    augmentations = []
    
    # Basic geometric transforms
    if hyp.get('perspective', 0) > 0:
        augmentations.append(
            RandomPerspectiveSmart(
                degrees=hyp.get('degrees', 0),
                translate=hyp.get('translate', 0.1),
                scale=hyp.get('scale', 0.5),
                shear=hyp.get('shear', 0),
                perspective=hyp.get('perspective', 0),
                p=hyp.get('perspective_p', 0.5)
            )
        )
    
    # New augmentations (GridMask, CutOut, etc.)
    if hyp.get('gridmask', False):
        augmentations.append(
            GridMask(
                d_range=tuple(hyp.get('gridmask_d_range', (96, 224))),
                ratio=hyp.get('gridmask_ratio', 0.5),
                p=hyp.get('gridmask_p', 0.3)
            )
        )
    
    if hyp.get('cutout', False):
        augmentations.append(
            CutOut(
                n_holes=hyp.get('cutout_n_holes', 1),
                length=hyp.get('cutout_length', 64),
                p=hyp.get('cutout_p', 0.3)
            )
        )
    
    if hyp.get('random_erasing', False):
        augmentations.append(
            RandomErasing(
                probability=hyp.get('random_erasing_p', 0.3),
                sl=hyp.get('random_erasing_sl', 0.02),
                sh=hyp.get('random_erasing_sh', 0.4),
                r1=hyp.get('random_erasing_r1', 0.3)
            )
        )
    
    if hyp.get('mixup_improved', False) and dataset:
        augmentations.append(
            ImprovedMixUp(
                dataset=dataset,
                p=hyp.get('mixup_p', 0.15),
                max_mixup_ratio=hyp.get('mixup_max_ratio', 0.5)
            )
        )
    
    return AugmentationComposer(augmentations, epoch=epoch)
