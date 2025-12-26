import os
import json
import sys
import torch
import torch.nn as nn
# 使用GL-MCM的自定义clip实现
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from clip import clip
from typing import Dict, List, Optional, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model_modular import build_modular_model


def load_clip_model(clip_ckpt: str = None, backbone: str = 'ViT-L/14', device: torch.device = None) -> nn.Module:
    """Load CLIP model with MCM compatibility."""
    # Use CLIP_ckpt if provided, otherwise use backbone
    clip_checkpoint = clip_ckpt if clip_ckpt else backbone
    
    # Load CLIP model
    clip_model, _ = clip.load(clip_checkpoint, device=device)
    
    # Ensure model is on the correct device
    clip_model = clip_model.to(device)
    
    return clip_model


def build_model_config(model_args: Dict, templates: List[str], device: torch.device) -> Dict:
    """Build model configuration dictionary."""
    cfg = {
        'device': device,
        'selector_type': model_args.get('selector_type'),
        'num_select': model_args.get('num_select'),
        'fuser_type': model_args.get('fuser_type'),
        'lambda_llm_negatives': model_args.get('lambda_llm_negatives'),
        'lambda_mixup': model_args.get('lambda_mixup'),
        'margin': model_args.get('margin', 0.2),
        'selector_temperature': model_args.get('selector_temperature', 1.0),
        'patches_per_slot_attn': model_args.get('patches_per_slot_attn', 16),
        'templates': templates,
        
        # Feature flags - Turn on everything for "Full Method"
        'use_redundancy_loss': True,
        'use_llm_negatives': True if model_args.get('lambda_llm_negatives', 0) > 0 else False,
        'use_semantic_exclusion': True,
        'use_mixup_invariance': True if model_args.get('lambda_mixup', 0) > 0 else False,
        
        # Loss weights
        'lambda_redundancy': 0.1,
    }
    return cfg


def build_modular_model_from_config(cfg: Dict, classnames: List[str], clip_model: nn.Module, 
                                   class_negatives: Optional[Dict] = None) -> nn.Module:
    """Build modular model from configuration."""
    model = build_modular_model(cfg, classnames, clip_model, class_negatives=class_negatives)
    return model.to(cfg['device'])


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all trainable parameters from the model."""
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params


def setup_optimizer_scheduler(trainable_params: List[nn.Parameter], lr: float, epochs: int) -> Tuple[torch.optim.SGD, CosineAnnealingLR]:
    """Setup optimizer and scheduler for training."""
    # Setup optimizer
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=0.9,
        weight_decay=1e-5
    )
    
    # Setup scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    return optimizer, scheduler
