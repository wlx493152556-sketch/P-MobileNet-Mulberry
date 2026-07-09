import os
# Use HF mirror for region-specific network optimization if needed
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import time
import copy
import random
import warnings
from glob import glob
from collections import Counter
from thop import profile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import io
from fvcore.nn import FlopCountAnalysis, parameter_count
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import timm  # Added: for loading comparison models
import torchvision.models as tv_models

warnings.filterwarnings('ignore')


# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
class Config:
    DATA_DIR = '/openbayes/input/input0/data cleaning'
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    EPOCHS = 300
    LR = 1e-3
    PATIENCE = 300
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WARMUP_EPOCHS = 10
    GRAD_CLIP_NORM = 1.0


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)


# ==========================================
# 2. Dataset Construction & Stratified Split
# ==========================================
class MulberryDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def preload_to_gpu(all_paths, all_labels):
    from PIL import Image
    from torchvision import transforms as T

    print("Preloading all images to GPU...")
    base_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    all_images = []
    for path in all_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = base_transform(img)
        all_images.append(img_tensor)

    all_images = torch.stack(all_images).to(Config.DEVICE)
    all_labels = torch.tensor(all_labels, dtype=torch.long).to(Config.DEVICE)
    print(f"Preloaded {len(all_images)} images to GPU, shape: {all_images.shape}")
    return all_images, all_labels


class GPUAugmentDataset(Dataset):
    """Dataset that performs data augmentation on GPU"""

    def __init__(self, images, labels, indices, transform_gpu=None):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform_gpu = transform_gpu

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img = self.images[actual_idx]
        label = self.labels[actual_idx]
        if self.transform_gpu:
            img = self.transform_gpu(img)
        return img, label


def prepare_dataloaders():
    classes = ['Disease Free leaves', 'Leaf Rust', 'Leaf spot']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    all_paths, all_labels = [], []
    for cls_name in classes:
        cls_dir = os.path.join(Config.DATA_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue
        paths = glob(os.path.join(cls_dir, '*.*'))
        all_paths.extend(paths)
        all_labels.extend([class_to_idx[cls_name]] * len(paths))

    print(f"Total images loaded: {len(all_paths)}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=Config.SEED
    )

    gpu_images, gpu_labels = preload_to_gpu(all_paths, all_labels)

    path_to_idx = {path: i for i, path in enumerate(all_paths)}
    temp_indices = [path_to_idx[p] for p in X_temp]
    test_indices = [path_to_idx[p] for p in X_test]

    counts = Counter(y_temp)
    class_weights = [1.0 / counts[i] for i in range(len(classes))]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)
    weights_tensor = weights_tensor / weights_tensor.sum() * len(classes)

    train_transform_gpu = transforms.Compose([
        transforms.RandomCrop((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform_gpu = transforms.Compose([
        transforms.CenterCrop((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(
        GPUAugmentDataset(gpu_images, gpu_labels, test_indices, val_test_transform_gpu),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )

    cv_data = (gpu_images, gpu_labels, temp_indices, y_temp,
               train_transform_gpu, val_test_transform_gpu)

    return cv_data, test_loader, weights_tensor


# ==========================================
# 3. Training & Evaluation Logic
# ==========================================
def train_model(model, train_loader, val_loader, weights_tensor, save_path):
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        if epoch <= Config.WARMUP_EPOCHS:
            lr = Config.LR * epoch / Config.WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)

        if epoch > Config.WARMUP_EPOCHS:
            scheduler.step()

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch:3d}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  -> Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return model


def measure_performance(model, test_loader, device='cuda'):
    """
    Comprehensive model performance measurement:
    - Classification metrics (Accuracy, Macro-P, Macro-R, Macro-F1)
    - Complexity metrics: Params(M), FLOPs(M), model file size (MB)
    - Inference speed metrics: latency (ms), FPS (using CUDA Event)
    - Resource usage: peak GPU memory (MB) during inference
    """
    # -------- Classification metrics (consistent with validation, using AMP) --------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.cuda.amp.autocast():  # ✅ AMP enabled
                out = model(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)

    acc = accuracy_score(all_labels, all_preds)
    mac_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # -------- Complexity metrics (independent of accuracy, no AMP needed) --------
    dummy = torch.randn(1, 3, 224, 224).to(device)
    params = parameter_count(model)[""]
    flops, _ = profile(model, inputs=(dummy,), verbose=False)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_mb = buffer.tell() / (1024 ** 2)

    # -------- Inference speed (CUDA Event + AMP) --------
    model.eval()
    x_single = torch.randn(1, 3, 224, 224).to(device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    timings = []
    warmup = 100
    repeat = 300
    with torch.no_grad():
        for i in range(warmup + repeat):
            if i == warmup:
                torch.cuda.synchronize()
            starter.record()
            with torch.cuda.amp.autocast():  # ✅ AMP enabled
                _ = model(x_single)
            ender.record()
            torch.cuda.synchronize()
            if i >= warmup:
                timings.append(starter.elapsed_time(ender))

    latency_ms = np.mean(timings)
    fps = 1000.0 / latency_ms

    # -------- Peak memory (AMP) --------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # ✅ AMP enabled
            _ = model(x_single)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        'Accuracy': acc,
        'Macro-Prec': mac_p,
        'Macro-Recall': mac_r,
        'Macro-F1': mac_f1,
        'Params(M)': params / 1e6,
        'FLOPs(M)': flops / 1e6,
        'Model Size(MB)': model_size_mb,
        'FPS': fps,
        'Latency(ms)': latency_ms,
        'Peak Memory(MB)': peak_mem_mb,
    }


# ==========================================
# 4. Core Experiment Controller — 5-Fold Cross-Validation (Comparison Models)
# ==========================================
def create_model(model_name, num_classes=3):
    """Unified model creation (without pretrained weights), tries timm first, then torchvision"""
    try:
        # Try loading from timm
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        print(f"✅ Loaded {model_name} from timm")
    except:
        try:
            # If timm fails, try loading from torchvision
            print(f"⚠️ timm failed, trying torchvision for {model_name}...")
            # Map model name to torchvision function name
            tv_name_map = {
                "shufflenet_v2_x0_5": "shufflenet_v2_x0_5",
                # "resnet18": "resnet18",
                # "mobilenet_v3_small": "mobilenet_v3_small",
            }
            func_name = tv_name_map.get(model_name, model_name)
            model = getattr(tv_models, func_name)(weights=None)

            # Modify classification head to 3 classes
            if hasattr(model, 'fc'):  # ResNet
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'classifier'):  # MobileNet/ShuffleNet
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'head'):  # Some ViT-like models
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
            print(f"✅ Loaded {model_name} from torchvision")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    return model


def run_experiment(model_name, loaders):
    """
    Parameters: loaders = (cv_data, test_loader, weights_tensor)
    Runs 5-fold cross-validation, trains one model per fold, evaluates on a fixed test set,
    and returns averaged metrics.
    """
    cv_data, test_loader, weights_tensor = loaders
    gpu_images, gpu_labels, temp_indices, y_temp, train_tf, val_tf = cv_data

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    fold_metrics = []

    print(f"\n{'=' * 70}\n🚀 Running Experiment: {model_name}\n{'=' * 70}")

    for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(temp_indices, y_temp)):
        save_path = f"best_{model_name.replace(' ', '_')}_fold{fold + 1}.pth"
        print(f"\n--- Fold {fold + 1}/5 ---")

        need_train = True
        if os.path.exists(save_path):
            print(f"✅ Found {save_path}, loading and evaluating...")
            try:
                model = create_model(model_name, num_classes=3).to(Config.DEVICE)
                model.load_state_dict(torch.load(save_path, map_location=Config.DEVICE))
                metrics = measure_performance(model, test_loader)
                del model
                torch.cuda.empty_cache()
                need_train = False
            except Exception as e:
                print(f"⚠️ Failed to load {save_path}: {e}. Removing and retraining...")
                os.remove(save_path)

        if need_train:
            fold_train_idx = [temp_indices[i] for i in train_fold_idx]
            fold_val_idx = [temp_indices[i] for i in val_fold_idx]

            train_loader = DataLoader(
                GPUAugmentDataset(gpu_images, gpu_labels, fold_train_idx, train_tf),
                batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
            )
            val_loader = DataLoader(
                GPUAugmentDataset(gpu_images, gpu_labels, fold_val_idx, val_tf),
                batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
            )

            model = create_model(model_name, num_classes=3).to(Config.DEVICE)
            model = train_model(model, train_loader, val_loader, weights_tensor, save_path)
            metrics = measure_performance(model, test_loader)
            del model
            torch.cuda.empty_cache()

        fold_metrics.append(metrics)

    # Average all classification metrics across folds and compute standard deviation
    avg_metrics = {}
    for key in ['Accuracy', 'Macro-Prec', 'Macro-Recall', 'Macro-F1']:
        vals = [m[key] for m in fold_metrics]
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        avg_metrics[key] = f"{mean_val:.4f}±{std_val:.4f}"

    # Structural/speed metrics are model-dependent, take values from the first fold
    for key in ['Params(M)', 'FLOPs(M)', 'Model Size(MB)', 'FPS', 'Latency(ms)', 'Peak Memory(MB)']:
        val = fold_metrics[0][key]
        avg_metrics[key] = f"{val:.4f}"

    print(f"\n✅ --- Test Results for {model_name} (averaged over 5 folds) ---")
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")

    return avg_metrics


if __name__ == "__main__":
    cv_data, test_loader, weights_tensor = prepare_dataloaders()
    loaders = (cv_data, test_loader, weights_tensor)

    # Comparison experiment model list (without pretrained weights)
    experiments = [
        "resnet50",
        "resnet18",
        "mobilenetv3_small_050",
        "ghostnet_050",
        "shufflenet_v2_x0_5",
        "fasternet_t0",
        "mobilevit_xxs",
        "mobilevitv2_050",
        "mobilenetv2_100",
        "mobilenetv2_050",
    ]

    all_results = []
    for model_name in experiments:
        res = run_experiment(model_name, loaders)
        res["Experiment"] = model_name
        all_results.append(res)
        torch.cuda.empty_cache()

    # Final comparison summary table
    col_width = 15
    header = ["Experiment", "Accuracy", "Macro-P", "Macro-R", "Macro-F1",
              "Params(M)", "FLOPs(M)", 'Model Size(MB)', "FPS", "Latency(ms)", 'Peak Memory(MB)']
    total_width = 22 + col_width * (len(header) - 1)

    print("\n\n" + "=" * total_width)
    print(f"{'🎯 COMPARISON EXPERIMENT SUMMARY (5-Fold CV) 🎯':^{total_width}}")
    print("=" * total_width)
    print(f"{header[0]:<22}" + "".join(f"{h:>{col_width}}" for h in header[1:]))
    print("-" * total_width)

    for res in all_results:
        vals = [res['Accuracy'], res['Macro-Prec'], res['Macro-Recall'], res['Macro-F1'],
                res['Params(M)'], res['FLOPs(M)'], res['Model Size(MB)'], res['FPS'], res['Latency(ms)'],
                res['Peak Memory(MB)']]
        print(f"{res['Experiment']:<22}" + "".join(f"{v:>{col_width}}" for v in vals))
    print("=" * total_width)