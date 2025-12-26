"""
Unified Training and Evaluation Script with Per-Epoch OOD Testing
Trains modular models with automatic ImageNet validation and OOD evaluation on each epoch.
Optimized for Linux servers with checkpoint savings for learnable parameters only.
Now supports Automatic Mixed Precision (AMP) for stability and speed.
"""

# imagenet_templates = [
#     'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
#     'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
#     'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
#     'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
#     'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.',
#     'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.',
#     'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
#     'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
#     'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
#     'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.',
#     'a photo of the dirty {}.', 'a jpeg of a {}.', 'a blurry photo of the {}.',
#     'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
#     'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.',
#     'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
#     'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
#     'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.',
#     'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.',
#     'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
#     'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.',
#     'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.',
#     'itap of the {}.', 'a jpeg of the {}.', 'a good photo of a {}.',
#     'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.',
#     'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.',
#     'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
#     'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
#     'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
#     'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
# ]
imagenet_templates = [
    'a photo of the {}.'
]
# �?_cache_text_features 中，你的代码已经写好�?mean() 逻辑�?
# 只要传入这个长列表，性能马上回升 3-5 个点�?


import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast # <--- 关键引入：自动混合精�?
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import clip
from .utils import Logger, cls_acc
from .data_utils import get_train_transform, get_test_transform, load_class_negatives, setup_mcm_data_loaders, setup_mixed_data_loaders
from .model_utils import load_clip_model, build_model_config, build_modular_model_from_config, get_trainable_params, setup_optimizer_scheduler
import math

class TrainEvalOrchestrator:
    """Orchestrates training and evaluation with per-epoch OOD testing."""
    
    def __init__(self, method: str, epochs: int, lr: float, 
                 batch_size: int, seed: int, device: torch.device,
                 selector_type: str = None, fuser_type: str = None, id_dataset: str = 'imagenet',
                 root_path: str = './data', shots: int = 16, lambda_llm_negatives: float = 0.1,
                 lambda_mixup: float = 0.1, margin: float = 0.2, num_select: int = 16,
                 backbone: str = 'ViT-L/14', class_negatives_path: str = '', use_full_data: bool = False,
                 # 新增参数用于 Slot/STE 调优
                 selector_temperature: float = 1.0,
                 patches_per_slot_attn: int = 16,
                 # OOD score parameters
                 score_type: str = 'GL-MCM',
                 temperature: float = 1.0,
                 lambda_local: float = 1.0,
                 # MCM compatibility parameters
                 in_dataset: str = None,
                 root_dir: str = None,
                 use_mcm_data_loader: bool = False,
                 CLIP_ckpt: str = None,
                 score: str = None,
                 num_ood_sumple: int = -1):
        
        self.method = method
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        
        # Model components
        self.selector_type = selector_type
        self.fuser_type = fuser_type
        
        # Dataset settings
        self.id_dataset = id_dataset
        self.root_path = root_path
        self.shots = shots
        self.use_full_data = use_full_data
        
        # Loss function coefficients (hyperparameters)
        self.lambda_llm_negatives = lambda_llm_negatives
        self.lambda_mixup = lambda_mixup
        self.margin = margin
        
        # OOD score parameters
        self.score_type = score_type
        self.temperature = temperature
        self.lambda_local = lambda_local
        
        # Model settings
        self.num_select = num_select
        self.backbone = backbone
        self.class_negatives_path = class_negatives_path
        self.class_negatives = None  # Initialize class_negatives to None
        
        # Advanced settings
        self.selector_temperature = selector_temperature
        self.patches_per_slot_attn = patches_per_slot_attn
        
        # MCM compatibility settings
        self.in_dataset = in_dataset
        self.root_dir = root_dir
        self.use_mcm_data_loader = use_mcm_data_loader
        self.CLIP_ckpt = CLIP_ckpt
        self.score = score
        self.num_ood_sumple = num_ood_sumple
        
        # Handle parameter precedence for compatibility
        if self.score is not None:
            self.score_type = self.score
        
        if self.root_dir is not None:
            self.root_path = self.root_dir
        
        if self.in_dataset is not None:
            self.id_dataset = self.in_dataset
        
        if self.score_type is not None:
            self.logger.log(f"Using OOD score type: {self.score_type}") if hasattr(self, 'logger') else None
        
        # Setup random seeds
        self._setup_seed(seed)
        
        # Classnames will be set in setup_data
        self.classnames = []
        
        # Setup logging
        self.log_dir = self._setup_logging()
        self.logger = Logger(os.path.join(self.log_dir, 'training.log'))
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None # For AMP
        self.train_loader = None
        self.test_loader = None
        self.ood_loaders = {}
        self.classnames = []
        
    def _setup_seed(self, seed: int):
        """Setup random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Fix for reproducibility

    def _setup_logging(self) -> str:
        """Setup logging directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'results/{self.method}_{self.selector_type}_{self.seed}_{timestamp}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{log_dir}/vis', exist_ok=True) # For visualizations
        return log_dir
    
    def setup_data(self):
        """Setup all data loaders (train, ID test, OOD tests) compatible with both FA and MCM."""
        self.logger.log('Setting up data...')
        
        # Use mixed data loading: FA for training, MCM for testing/OOD
        self.logger.log("Using mixed data loading: FA for training, MCM for testing/OOD")
        
        # Setup data loaders using mixed style
        data_loaders, ood_loaders, classnames, class_negatives = setup_mixed_data_loaders(
            root_path=self.root_path,
            batch_size=self.batch_size,
            id_dataset=self.id_dataset,
            shots=self.shots,
            use_full_data=self.use_full_data,
            num_ood_sumple=self.num_ood_sumple,
            logger=self.logger,
            class_negatives_path=self.class_negatives_path
        )
        
        # Extract loaders
        self.train_loader = data_loaders['train']
        self.test_loader = data_loaders['test']
        self.ood_loaders = ood_loaders
        self.classnames = classnames
        
        # Update class negatives if loaded from data loader
        if class_negatives:
            self.class_negatives = class_negatives
    
    
    def setup_model(self):
        """Initialize model, optimizer, and scheduler compatible with MCM."""
        self.logger.log('Setting up model...')
        
        # Load CLIP model (MCM style)
        clip_model = load_clip_model(
            clip_ckpt=self.CLIP_ckpt,
            backbone=self.backbone,
            device=self.device
        )
        
        # Create model configuration
        model_args = {
            'selector_type': self.selector_type,
            'num_select': self.num_select,
            'fuser_type': self.fuser_type,
            'lambda_llm_negatives': self.lambda_llm_negatives,
            'lambda_mixup': self.lambda_mixup,
            'margin': self.margin,
            'selector_temperature': self.selector_temperature,
            'patches_per_slot_attn': self.patches_per_slot_attn
        }
        
        cfg = build_model_config(model_args, imagenet_templates, self.device)
        self.logger.log(f'Config: {json.dumps(cfg, default=str, indent=2)}')
        
        # Build modular model
        self.model = build_modular_model_from_config(
            cfg=cfg,
            classnames=self.classnames,
            clip_model=clip_model,
            class_negatives=self.class_negatives
        )
        
        # Setup optimizer and scheduler
        trainable_params = get_trainable_params(self.model)
        
        if not trainable_params:
            self.logger.log("WARNING: No trainable parameters found! Check your model configuration.")
        
        self.optimizer, self.scheduler = setup_optimizer_scheduler(
            trainable_params=trainable_params,
            lr=self.lr,
            epochs=self.epochs
        )
        
        # Setup GradScaler for AMP
        self.scaler = GradScaler()
        
        self.logger.log(f'  ✓ Model: {self.method}')
        self.logger.log(f'  ✓ Trainable parameters: {sum(p.numel() for p in trainable_params)}')
    
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with AMP (Automatic Mixed Precision)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Tqdm bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]', ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # 1. Prepare Data
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            negative_text_tokens = None  # No longer needed since we use cached negative features
            
            self.optimizer.zero_grad()
            
            # 2. Forward with Autocast (Mixed Precision) - 关键修改
            # 模型的大部分计算(Conv, Linear)在FP16下进行，Loss计算会自�?通过装饰�?切回FP32
            with autocast():
                output_dict = self.model(images, labels=labels, negative_text_tokens=negative_text_tokens)
                
                logits = output_dict['logits']
                aux_losses = output_dict['aux_losses']
                
                # Main Loss
                ce_loss = F.cross_entropy(logits, labels)
                
                # Aux Losses
                total_aux_loss = 0.0
                for k, v in aux_losses.items():
                    if v.requires_grad:
                        total_aux_loss += v
                
                loss = ce_loss + total_aux_loss

            # 3. Backward with Scaler - 关键修改
            # Scaler 会自动处理梯度下�?underflow)问题，无需手动检�?NaN
            self.scaler.scale(loss).backward()

            # 4. Gradient Clipping (must unscale first)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 5. Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 6. Metrics & Logging
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Prepare loss info for display
            loss_info = {
                'Loss': f"{loss.item():.4f}",
                'CE': f"{ce_loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            }
            
            # Add auxiliary losses to display
            for k, v in aux_losses.items():
                if v.requires_grad:
                    loss_info[k] = f"{v.item():.4f}"
            
            # Update progress bar with detailed loss info
            pbar.set_postfix(loss_info)
            
            # Log detailed loss info occasionally
            if batch_idx % 50 == 0:
                loss_str = f"CE: {ce_loss.item():.4f}"
                for k, v in aux_losses.items():
                    loss_str += f", {k}: {v.item():.4f}"
                self.logger.log(f"  Batch {batch_idx}: {loss_str}")

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        train_acc = 100.0 * correct / total if total > 0 else 0
        
        return avg_loss, train_acc, True
    
    def evaluate_id(self) -> float:
        """Evaluate on ID (ImageNet) test set."""

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Inference only needs images (labels are for metric calc only)
                with autocast(): # 推理也使用FP16加�?
                    output_dict = self.model(images, labels=None) # No labels passed = Inference mode
                    logits = output_dict['logits']
                
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        id_acc = 100.0 * correct / total if total > 0 else 0.0
        return id_acc
    
    def evaluate_ood_dataset(self, dataset_name: str) -> Tuple[float, float]:
        """Evaluate on single OOD dataset."""
        if dataset_name not in self.ood_loaders:
            return 0.0, 0.0
        
        self.model.eval()
        
        # 1. Get ID confidences
        to_np = lambda x: x.data.cpu().numpy()
        id_scores = []
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                with autocast():
                    output_dict = self.model(images)
                                        
                    # Extract features similar to get_ood_scores_clip
                    global_features = output_dict['global_features']
                    local_features = output_dict['local_features']
                    text_feats = self.model._text_features.to(self.device)
                    
                    # Normalize features
                    # global_features = F.normalize(global_features, dim=-1)
                    global_features = global_features / global_features.norm(dim=-1, keepdim=True)
                    # local_features = F.normalize(local_features, dim=-1)
                    local_features = local_features / local_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate logits (no logit_scale, consistent with GL-MCM)
                    output_global = global_features @ text_feats.T
                    output_local = local_features @ text_feats.T
                    
                    # Calculate scores based on score_type (consistent with GL-MCM)
                    if self.score_type == 'energy':
                        # Energy = - T * logsumexp(logit_k / T)
                        output_global_np = to_np(output_global)
                        scores = -self.temperature * np.logaddexp.reduce(output_global_np / self.temperature, axis=1)
                    elif self.score_type == 'entropy':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        scores = entropy(smax_global, axis=1)
                    elif self.score_type == 'var':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        scores = -np.var(smax_global, axis=1)
                    elif self.score_type == 'MCM':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        scores = -np.max(smax_global, axis=1)
                    elif self.score_type == 'max-logit':
                        output_global_np = to_np(output_global)
                        scores = -np.max(output_global_np, axis=1)
                    elif self.score_type == 'L-MCM':
                        # Local MCM - calculate softmax over local features
                        smax_local = to_np(F.softmax(output_local / self.temperature, dim=1))
                        # For ViT, local_features shape is (B, N, D) where N is number of patches
                        # We need to reshape to match GL-MCM's expected shape (B, H, W, C) for spatial dimensions
                        B, N, C = smax_local.shape
                        # Assume square patches
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        # Get max over spatial dimensions (H, W)
                        scores = -np.max(smax_local_reshaped, axis=(1, 2, 3))
                    elif self.score_type == 'GL-MCM':
                        # Global-Local MCM combination
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        mcm_global_score = -np.max(smax_global, axis=1)
                        
                        # Local MCM component
                        smax_local = to_np(F.softmax(output_local / self.temperature, dim=1))
                        B, N, C = smax_local.shape
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        mcm_local_score = -np.max(smax_local_reshaped, axis=(1, 2, 3))
                        
                        # Combine with lambda_local weight
                        scores = mcm_global_score + self.lambda_local * mcm_local_score
                
                
                id_scores.extend(scores)
        
        # 2. Get OOD confidences
        ood_scores = []
        with torch.no_grad():
            for batch in self.ood_loaders[dataset_name]:
                if isinstance(batch, dict):
                    images = batch['images'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                with autocast():
                    output_dict = self.model(images)
                                        
                    # Extract features similar to get_ood_scores_clip
                    global_features = output_dict['global_features']
                    local_features = output_dict['local_features']
                    text_feats = self.model._text_features.to(self.device)
                    
                    # Normalize features (consistent with GL-MCM)
                    global_features = global_features / global_features.norm(dim=-1, keepdim=True)
                    local_features = local_features / local_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate logits (no logit_scale, consistent with GL-MCM)
                    output_global = global_features @ text_feats.T
                    output_local = local_features @ text_feats.T
                    
                    # Calculate scores based on score_type (consistent with GL-MCM)
                    if self.score_type == 'energy':
                        # Energy = - T * logsumexp(logit_k / T)
                        output_global_np = to_np(output_global)
                        scores = -self.temperature * np.logaddexp.reduce(output_global_np / self.temperature, axis=1)
                    elif self.score_type == 'entropy':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        scores = entropy(smax_global, axis=1)
                    elif self.score_type == 'var':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        scores = -np.var(smax_global, axis=1)
                    elif self.score_type == 'MCM':
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        scores = -np.max(smax_global, axis=1)
                    elif self.score_type == 'max-logit':
                        output_global_np = to_np(output_global)
                        scores = -np.max(output_global_np, axis=1)
                    elif self.score_type == 'L-MCM':
                        # Local MCM - calculate softmax over local features
                        smax_local = to_np(F.softmax(output_local / self.temperature, dim=1))
                        # For ViT, local_features shape is (B, N, D) where N is number of patches
                        # We need to reshape to match GL-MCM's expected shape (B, H, W, C) for spatial dimensions
                        B, N, C = smax_local.shape
                        # Assume square patches
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        # Get max over spatial dimensions (H, W)
                        scores = -np.max(smax_local_reshaped, axis=(1, 2, 3))
                    elif self.score_type == 'GL-MCM':
                        # Global-Local MCM combination
                        smax_global = to_np(F.softmax(output_global / self.temperature, dim=1))
                        mcm_global_score = -np.max(smax_global, axis=1)
                        
                        # Local MCM component
                        smax_local = to_np(F.softmax(output_local / self.temperature, dim=1))
                        B, N, C = smax_local.shape
                        H = W = int(math.sqrt(N))
                        smax_local_reshaped = smax_local.reshape(B, H, W, C)
                        mcm_local_score = -np.max(smax_local_reshaped, axis=(1, 2, 3))
                        
                        # Combine with lambda_local weight
                        scores = mcm_global_score + self.lambda_local * mcm_local_score
                
                
                ood_scores.extend(scores)
        
        # 3. Compute metrics
        if len(id_scores) == 0 or len(ood_scores) == 0:
            return 0.0, 0.0
            
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        
        # 使用负号处理，与GL-MCM保持一�?
        auroc = self._compute_auroc(-id_scores, -ood_scores)
        fpr95 = self._compute_fpr95(-id_scores, -ood_scores)
        
        return auroc, fpr95
    
    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)
        expected = np.sum(arr, dtype=np.float64)
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                               'its last element does not correspond to sum')
        return out
    
    def fpr_and_fdr_at_recall(self, y_true, y_score, recall_level=0.95, pos_label=None):
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                         np.array_equal(classes, [-1, 1]) or
                         np.array_equal(classes, [0]) or
                         np.array_equal(classes, [-1]) or
                         np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.
    
        # make y_true a boolean vector
        y_true = (y_true == pos_label)
    
        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
    
        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
        # accumulate the true positives with decreasing threshold
        tps = self.stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing
    
        thresholds = y_score[threshold_idxs]
    
        recall = tps / tps[-1]
    
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)      # [last_ind::-1]
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    
        cutoff = np.argmin(np.abs(recall - recall_level))
    
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])
    
    def get_measures(self, id_scores: np.ndarray, ood_scores: np.ndarray, recall_level=0.95):
        """Calculate AUROC, AUPR and FPR95 using GL-MCM's method"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        pos = np.array(id_scores[:]).reshape((-1, 1))
        neg = np.array(ood_scores[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1
    
        try:
            auroc = roc_auc_score(labels, examples)
            aupr = average_precision_score(labels, examples)
            fpr = self.fpr_and_fdr_at_recall(labels, examples, recall_level)
        except:
            auroc, aupr, fpr = 0.0, 0.0, 0.0
        
        return auroc, aupr, fpr
    
    def _compute_auroc(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        """Compute AUROC (wrapped for backward compatibility)"""
        auroc, _, _ = self.get_measures(id_scores, ood_scores)
        return auroc * 100
    
    def _compute_fpr95(self, id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        """Compute FPR95 (wrapped for backward compatibility)"""
        _, _, fpr = self.get_measures(id_scores, ood_scores, recall_level=0.95)
        return fpr * 100
    
    def evaluate_epoch(self, epoch: int) -> Dict:
        """Evaluate on ID and all OOD datasets."""
        results = {}
        
        # ID evaluation
        id_acc = self.evaluate_id()
        results['id_accuracy'] = id_acc
        
        # OOD evaluations
        ood_aurocs = []
        ood_fpr95s = []
        
        for ood_name in self.ood_loaders.keys():
            auroc, fpr95 = self.evaluate_ood_dataset(ood_name)
            results[f'{ood_name}_auroc'] = auroc
            results[f'{ood_name}_fpr95'] = fpr95
            ood_aurocs.append(auroc)
            ood_fpr95s.append(fpr95)
        
        # Averages
        results['avg_ood_auroc'] = np.mean(ood_aurocs) if ood_aurocs else 0.0
        results['avg_ood_fpr95'] = np.mean(ood_fpr95s) if ood_fpr95s else 0.0
        
        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save checkpoint with only trainable parameters."""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'epoch_{epoch:03d}.pt')
        
        # Extract only trainable parameters
        trainable_state = {}
        for name, param in self.model.named_parameters():
            # Save if requires grad OR if it's a buffer like running_mean
            if param.requires_grad:
                trainable_state[name] = param.data.clone()
        
        checkpoint = {
            'epoch': epoch,
            'method': self.method,
            'seed': self.seed,
            'state_dict': trainable_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(), # Save scaler state
            'metrics': metrics,
            'config': {
                'selector_type': self.selector_type,
                'fuser_type': self.fuser_type,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def save_visualization(self, epoch):
        pass

    def train_with_eval(self):
        """Main training loop."""
        self.logger.log('\n' + '='*80)
        self.logger.log(f'Starting training: {self.method} (seed={self.seed})')
        self.logger.log('='*80 + '\n')
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        best_avg_auroc = 0.0
        results_history = []
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc, _ = self.train_epoch(epoch)
            self.scheduler.step()
            

            eval_results={}
            
            # Log results
            self.logger.log(f'\n[Epoch {epoch+1}/{self.epochs}]')
            self.logger.log(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            if (epoch+1)%5==0 and epoch+1>0:
                # Evaluate
                eval_results = self.evaluate_epoch(epoch)
                self.logger.log(f'  ID Accuracy: {eval_results["id_accuracy"]:.2f}%')
                
                for ood_name in self.ood_loaders.keys():
                    auroc = eval_results[f'{ood_name}_auroc']
                    fpr95 = eval_results[f'{ood_name}_fpr95']
                    self.logger.log(f'  {ood_name:15} AUROC: {auroc:.2f}%, FPR95: {fpr95:.2f}%')
                
                avg_auroc = eval_results["avg_ood_auroc"]
                avg_fpr95 = eval_results["avg_ood_fpr95"]
                self.logger.log(f'  Avg OOD AUROC: {avg_auroc:.2f}%, Avg OOD FPR95: {avg_fpr95:.2f}%')
            
                # Save checkpoint (Every 5 epochs or best)
                if (epoch % 5 == 0 or epoch == self.epochs - 1) and epoch+1>10:
                    self.save_checkpoint(epoch, eval_results)
            
                # Track best
                if avg_auroc > best_avg_auroc:
                    best_avg_auroc = avg_auroc
                    self.save_checkpoint(999, eval_results) # 999 as code for 'best'
                    self.logger.log(f"  �?New Best Avg AUROC: {best_avg_auroc:.2f}%")
            
            # Update history
            results_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                **eval_results
            })
            
            # Dump history
            with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
                json.dump(results_history, f, indent=2)
                
        self.logger.log('\nTraining completed.')


def main():
    parser = argparse.ArgumentParser(description='Train modular OOD detection with AMP')
    
    # Basic Config
    parser.add_argument('--method', type=str, default='custom')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    # Components
    parser.add_argument('--selector_type', type=str, default='slot', help="'mlp' or 'slot'")
    parser.add_argument('--fuser_type', type=str, default='query_attn')
    
    # Dataset
    parser.add_argument('--id_dataset', type=str, default='imagenet', choices=['ImageNet', 'COCO_single', 'COCO_multi', 'VOC_single'])
    parser.add_argument('--in_dataset', type=str, default=None, help='MCM compatibility for in-distribution dataset')
    parser.add_argument('--root_path', type=str, default='/data/datasets')
    parser.add_argument('--root-dir', type=str, default=None, help='MCM compatibility for root directory')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--use_full_data', action='store_true')
    parser.add_argument('--use_mcm_data_loader', action='store_true', help='Use MCM-style data loading')
    
    # Loss Weights
    parser.add_argument('--lambda_llm_negatives', type=float, default=0.1)
    parser.add_argument('--lambda_mixup', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.2)
    
    # Paths
    parser.add_argument('--class_negatives_path', type=str, default='configs/class_negatives_clean.json')
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--CLIP_ckpt', type=str, default=None, help='MCM compatibility for CLIP checkpoint')
    
    # OOD scoring parameters (MCM compatible)
    parser.add_argument('--score', type=str, default='GL-MCM', choices=['MCM', 'L-MCM', 'GL-MCM'], help='OOD score options')
    parser.add_argument('--score_type', type=str, default=None, help='OOD score type for compatibility')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for OOD scoring')
    parser.add_argument('--lambda_local', type=float, default=1.0, help='Weight for local score in GL-MCM')
    parser.add_argument('--num_ood_sumple', default=-1, type=int, help='Numbers of OOD samples')

    # === 关键修正：确保包�?num_select 参数 ===
    parser.add_argument('--num_select', type=int, default=16,
                        help='Number of tokens/features to retain in selector (k)')
    # ========================================

    # Advanced params
    parser.add_argument('--selector_temperature', type=float, default=1.0)
    parser.add_argument('--patches_per_slot_attn', type=int, default=16)

    args = parser.parse_args()

    # Create trainer
    trainer = TrainEvalOrchestrator(
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=torch.device(args.device),
        selector_type=args.selector_type,
        fuser_type=args.fuser_type,
        id_dataset=args.id_dataset,
        root_path=args.root_path,
        shots=args.shots,
        lambda_llm_negatives=args.lambda_llm_negatives,
        lambda_mixup=args.lambda_mixup,
        margin=args.margin,
        backbone=args.backbone,
        class_negatives_path=args.class_negatives_path,
        use_full_data=args.use_full_data,
        num_select=args.num_select,
        selector_temperature=args.selector_temperature,
        patches_per_slot_attn=args.patches_per_slot_attn,
        # MCM compatibility parameters
        in_dataset=args.in_dataset,
        root_dir=args.root_dir,
        use_mcm_data_loader=args.use_mcm_data_loader,
        CLIP_ckpt=args.CLIP_ckpt,
        score=args.score,
        num_ood_sumple=args.num_ood_sumple
    )

    trainer.train_with_eval()


if __name__ == '__main__':
    main()
