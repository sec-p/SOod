import os
import json
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from typing import Dict, List, Tuple, Optional

# Import FA data loading functions
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
try:
    from my_dataset import build_dataset
    from my_dataset.utils import build_data_loader
except ImportError:
    # Fallback if my_dataset is not available
    build_dataset = None
    build_data_loader = None


def get_train_transform():
    """Get training transforms."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1),
                                    interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])


def get_test_transform():
    """Get test transforms."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])



def load_class_negatives(class_negatives_path: str, root_path: str, logger=None) -> Dict:
    """Load class negatives from file with fallback paths."""
    class_negatives = {}
    neg_path = class_negatives_path
    if neg_path and os.path.exists(neg_path):
        with open(neg_path, 'r') as f:
            class_negatives = json.load(f)
        return class_negatives
    
    # Fallback to default locations
    fallback = os.path.join(os.path.dirname(__file__), 'my_dataset', 'class_negatives.json')
    fallback_src = os.path.join(os.path.dirname(__file__), '../configs', 'class_negatives.json')
    
    if os.path.exists(fallback):
        with open(fallback, 'r') as f:
            return json.load(f)
    elif os.path.exists(fallback_src):
        with open(fallback_src, 'r') as f:
            return json.load(f)
            
    return class_negatives


def load_imagenet_classnames(logger=None) -> List[str]:
    """Load ImageNet class names from GL-MCM's data files."""
    # Try to load from imagenet_class_clean.npy first
    class_clean_path = os.path.join(os.path.dirname(__file__), '../data/ImageNet/imagenet_class_clean.npy')
    if os.path.exists(class_clean_path):
        try:
            classnames = np.load(class_clean_path, allow_pickle=True).tolist()
            if isinstance(classnames, str):
                # If it's a string with spaces between class names
                classnames = classnames.split()
            if logger:
                logger.log(f"✓ Loaded {len(classnames)} class names from {class_clean_path}")
            return classnames
        except Exception as e:
            if logger:
                logger.log(f"⚠ Failed to load class names from {class_clean_path}: {e}")
    
    # Fallback to imagenet_class_index.json
    class_index_path = os.path.join(os.path.dirname(__file__), '../data/ImageNet/imagenet_class_index.json')
    if os.path.exists(class_index_path):
        try:
            with open(class_index_path, 'r') as f:
                class_index = json.load(f)
            # Extract just the class names (second element in each value)
            classnames = [class_info[1] for class_info in class_index.values()]
            if logger:
                logger.log(f"✓ Loaded {len(classnames)} class names from {class_index_path}")
            return classnames
        except Exception as e:
            if logger:
                logger.log(f"⚠ Failed to load class names from {class_index_path}: {e}")
    
    if logger:
        logger.log("⚠ Failed to load GL-MCM class names, will use default ImageFolder class names")
    
    return []



def get_subset_with_len(dataset, length, shuffle=False):
    """Get a subset of the dataset with the specified length."""
    dataset_size = len(dataset)
    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length
    return subset



def setup_mixed_data_loaders(root_path: str, batch_size: int, id_dataset: str = "ImageNet", 
                            shots: int = 16, use_full_data: bool = False, 
                            num_ood_sumple: int = -1, logger=None) -> Tuple[Dict, Dict, List, Optional[Dict]]:
    """Setup mixed data loaders with FA-style training data and MCM-style test/OOD data."""
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    
    data_loaders = {}
    ood_loaders = {}
    class_negatives = {}
    
    # Try to load GL-MCM's ImageNet class names first
    gl_mcm_classnames = load_imagenet_classnames(logger)
    
    # Setup training data loader using FA's implementation
    if build_dataset is not None and build_data_loader is not None:
        try:
            shots = -1 if use_full_data else shots
            id_dataset_obj = build_dataset(id_dataset, root_path, shots)
            
            # Get training data and classnames
            train_data = id_dataset_obj.train_x
            
            # Use GL-MCM class names if available, otherwise use FA's class names
            if gl_mcm_classnames:
                classnames = gl_mcm_classnames
            else:
                classnames = id_dataset_obj.classnames
            
            # Load class negatives if available
            class_negatives = load_class_negatives("", root_path, logger)
            if class_negatives and logger:
                logger.log(f"Loaded class negatives for {len(class_negatives)} classes")
            
            # Build training data loader
            data_loaders['train'] = build_data_loader(
                data_source=train_data,
                batch_size=batch_size,
                tfm=train_transform,
                is_train=True,
                shuffle=True
            )
            
            if logger:
                logger.log(f"✓ Using FA's data loader for training: {len(train_data)} samples")
        except Exception as e:
            if logger:
                logger.log(f"⚠ Failed to use FA's data loader for training: {e}")
                logger.log("  Falling back to MCM's ImageFolder implementation")
            
            # Fallback to MCM's ImageFolder implementation for training
            train_dataset = datasets.ImageFolder(
                os.path.join(root_path, 'ImageNet'), 
                transform=train_transform
            )
            
            data_loaders['train'] = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            # Use GL-MCM class names if available, otherwise use ImageFolder's class names
            if gl_mcm_classnames:
                classnames = gl_mcm_classnames
            else:
                classnames = train_dataset.classes
    else:
        # Fallback to MCM's ImageFolder implementation for training
        if logger:
            logger.log("⚠ FA's data loader not available, using MCM's ImageFolder implementation")
        
        train_dataset = datasets.ImageFolder(
            os.path.join(root_path, 'ImageNet'), 
            transform=train_transform
        )
        
        data_loaders['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Use GL-MCM class names if available, otherwise use ImageFolder's class names
        if gl_mcm_classnames:
            classnames = gl_mcm_classnames
        else:
            classnames = train_dataset.classes
    
    # Setup test data loader using MCM's ImageFolder implementation
    test_dataset = datasets.ImageFolder(
        os.path.join(root_path, 'ImageNet'), 
        transform=test_transform
    )
    
    data_loaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup OOD data loaders using MCM's ImageFolder implementation
    OOD_DATASETS = ['iNaturalist', 'SUN', 'Places', 'Textures']
    for ood_dataset_name in OOD_DATASETS:
        try:
            ood_path = os.path.join(root_path, ood_dataset_name)
            if not os.path.exists(ood_path):
                # Try alternative naming for Textures
                if ood_dataset_name == 'Textures':
                    ood_path = os.path.join(root_path, 'Texture', 'images')
                else:
                    continue
            
            ood_dataset = datasets.ImageFolder(
                root=ood_path,
                transform=test_transform
            )
            
            # Apply subset sampling if specified
            if num_ood_sumple > 0:
                ood_dataset = get_subset_with_len(ood_dataset, length=num_ood_sumple, shuffle=True)
            
            ood_loader = torch.utils.data.DataLoader(
                ood_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            ood_loaders[ood_dataset_name] = ood_loader
            if logger:
                logger.log(f'  ✓ Loaded OOD dataset: {ood_dataset_name}')
        except Exception as e:
            if logger:
                logger.log(f'  ⚠ Failed to load OOD dataset {ood_dataset_name}: {e}')
    
    return data_loaders, ood_loaders, classnames, class_negatives


def setup_mcm_data_loaders(root_path: str, batch_size: int, use_mcm_data_loader: bool = False, 
                          id_dataset: str = "ImageNet", num_ood_sumple: int = -1, logger=None) -> Tuple[Dict, Dict, List]:
    """Setup MCM-style data loaders for training, ID testing, and OOD testing (deprecated)."""
    # Call the new mixed data loaders function with default parameters
    data_loaders, ood_loaders, classnames, _ = setup_mixed_data_loaders(
        root_path, batch_size, id_dataset, num_ood_sumple=num_ood_sumple, logger=logger
    )
    
    return data_loaders, ood_loaders, classnames
