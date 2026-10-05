import os
import math
import time
import copy
import random
import io
import warnings
from glob import glob
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from fvcore.nn import FlopCountAnalysis, parameter_count

warnings.filterwarnings('ignore')


# ==========================================
# 1. Hyperparameters and configuration
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
# 2. Dataset construction and stratified splitting
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
    """Dataset with GPU-side data augmentation"""

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
    """Return materials required for cross-validation"""
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
# 3. Improved model architecture
# ==========================================
class PConv(nn.Module):
    def __init__(self, dim, ratio=0.25):
        super().__init__()
        self.dim_conv = max(1, int(dim * ratio))
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.act(self.bn(self.conv(x1)))
        return torch.cat((x1, x2), dim=1)


class ECA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        k = int(abs((math.log2(channels) / 2) + 0.5))
        k = k if k % 2 == 1 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_pconv=False, use_eca=False):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and inp == oup)
        hidden_dim = int(round(inp * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        if use_pconv and stride == 1:
            layers.append(PConv(hidden_dim, ratio=0.25))
        else:
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                                    groups=hidden_dim, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        if use_eca:
            layers.append(ECA(hidden_dim))

        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class ProposedMobileNetV2(nn.Module):
    def __init__(self, num_classes=3, width_mult=0.25, expand_ratio=6, use_pconv=False, use_eca=False):
        super().__init__()
        settings = [[1, 16, 1, 1],
                    [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
                    [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
                    ]

        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v: new_v += divisor
            return new_v

        input_channel = _make_divisible(32 * width_mult)
        self.last_channel = _make_divisible(1280 * max(1.0, width_mult))

        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        for t, c, n, s in settings:
            output_channel = _make_divisible(c * width_mult)
            t_actual = t if t == 1 else expand_ratio
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(
                    input_channel, output_channel, stride, t_actual,
                    use_pconv=use_pconv, use_eca=use_eca
                ))
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, self.last_channel, 1, 1, bias=False))
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


# ==========================================
# 4. Training and evaluation logic
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
    Comprehensive model performance measurement (using AMP for consistency):
    - Classification metrics: Accuracy, Macro-P, Macro-R, Macro-F1
    - Complexity metrics: Params(M), FLOPs(M), model file size(MB)
    - Inference speed: Latency(ms), FPS (using CUDA Event)
    - Resource usage: Peak GPU memory(MB) during inference
    """
    # -------- Classification metrics (using AMP consistent with validation) --------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.cuda.amp.autocast():
                out = model(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)

    acc = accuracy_score(all_labels, all_preds)
    mac_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # -------- Complexity metrics (using fvcore) --------
    dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    
    # Parameter count
    params = parameter_count(model)[""]
    
    # FLOPs (fvcore directly outputs FLOPs, no multiplication by 2)
    flops = FlopCountAnalysis(model, dummy).total()
    
    # Model file size
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_mb = buffer.tell() / (1024 ** 2)

    # -------- Inference speed (CUDA Event + AMP) --------
    model.eval()
    x_single = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
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
            with torch.cuda.amp.autocast():
                _ = model(x_single)
            ender.record()
            torch.cuda.synchronize()
            if i >= warmup:
                timings.append(starter.elapsed_time(ender))
    
    latency_ms = np.mean(timings)
    fps = 1000.0 / latency_ms

    # -------- Peak GPU memory (AMP) --------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
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
# 5. Core experiment controller — 5-fold cross-validation with checkpoint resume
# ==========================================
def run_experiment(exp_name, t, use_pconv, use_eca, loaders):
    """
    Run 5-fold cross-validation, train one model per fold (supports checkpoint resume),
    evaluate on a fixed test set, return averaged metrics.
    """
    cv_data, test_loader, weights_tensor = loaders
    gpu_images, gpu_labels, temp_indices, y_temp, train_tf, val_tf = cv_data

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    fold_metrics = []

    print(f"\n{'='*70}\n🚀 Running Experiment: {exp_name}\n{'='*70}")

    for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(temp_indices, y_temp)):
        save_path = f"best_{exp_name.replace(' ', '_')}_fold{fold+1}.pth"
        print(f"\n--- Fold {fold+1}/5 ---")
        
        # ========== Checkpoint resume logic ==========
        model = ProposedMobileNetV2(
            num_classes=3, width_mult=0.25, expand_ratio=t,
            use_pconv=use_pconv, use_eca=use_eca
        ).to(Config.DEVICE)
        
        need_train = True
        if os.path.exists(save_path):
            print(f"📁 Found existing checkpoint: {save_path}")
            try:
                # Attempt to load existing weights
                checkpoint = torch.load(save_path, map_location=Config.DEVICE)
                model.load_state_dict(checkpoint)
                print(f"✅ Successfully loaded weights, skipping training for Fold {fold+1}")
                need_train = False
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint ({e}), will retrain...")
                os.remove(save_path)  # Delete corrupted file
        
        if need_train:
            print(f"🔄 Training Fold {fold+1}...")
            # Build DataLoader for this fold
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
            
            model = train_model(model, train_loader, val_loader, weights_tensor, save_path)
        
        # Unified evaluation
        metrics = measure_performance(model, test_loader)
        fold_metrics.append(metrics)
        
        # ========== Print per-fold metrics immediately ==========
        print(f"\n📊 Fold {fold+1} Test Metrics:")
        print(f"   Accuracy:    {metrics['Accuracy']:.4f}")
        print(f"   Macro-Prec:  {metrics['Macro-Prec']:.4f}")
        print(f"   Macro-Recall:{metrics['Macro-Recall']:.4f}")
        print(f"   Macro-F1:    {metrics['Macro-F1']:.4f}")
        print(f"   Params(M):   {metrics['Params(M)']:.4f}")
        print(f"   FLOPs(M):    {metrics['FLOPs(M)']:.4f}")
        print(f"   Model(MB):   {metrics['Model Size(MB)']:.4f}")
        print(f"   FPS:         {metrics['FPS']:.2f}")
        print(f"   Latency(ms): {metrics['Latency(ms)']:.2f}")
        print(f"   PeakMem(MB): {metrics['Peak Memory(MB)']:.4f}")
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Average classification metrics across folds with standard deviation
    avg_metrics = {}
    for key in ['Accuracy', 'Macro-Prec', 'Macro-Recall', 'Macro-F1']:
        vals = [m[key] for m in fold_metrics]
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        avg_metrics[key] = f"{mean_val:.4f}±{std_val:.4f}"

    # Architecture/speed metrics are fold-independent, take the first fold's value
    for key in ['Params(M)', 'FLOPs(M)', 'Model Size(MB)', 'Peak Memory(MB)']:
        val = fold_metrics[0][key]
        avg_metrics[key] = f"{val:.4f}"
    
    for key in ['FPS', 'Latency(ms)']:
        val = fold_metrics[0][key]
        avg_metrics[key] = f"{val:.2f}"

    print(f"\n✅ --- Test Results for {exp_name} (averaged over 5 folds) ---")
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")

    return avg_metrics


if __name__ == "__main__":
    cv_data, test_loader, weights_tensor = prepare_dataloaders()
    loaders = (cv_data, test_loader, weights_tensor)

    # Ablation experiment configuration
    # the ECA variant is kept only as an ablation entry.
    experiments = [
        {"name": "Baseline (t=6)",  "t": 6, "use_pconv": False, "use_eca": False},
        {"name": "t=6+PConv",       "t": 6, "use_pconv": True,  "use_eca": False},
        {"name": "t=3",             "t": 3, "use_pconv": False, "use_eca": False},
        {"name": "t=3+PConv",     "t": 3, "use_pconv": True,  "use_eca": False},
        {"name": "t=3+PConv+ECA",   "t": 3, "use_pconv": True,  "use_eca": True},
    ]

    all_results = []
    for exp in experiments:
        res = run_experiment(exp["name"], exp["t"], exp["use_pconv"], exp["use_eca"], loaders)
        res["Experiment"] = exp["name"]
        all_results.append(res)
        torch.cuda.empty_cache()

    # Final comparison table
    col_width = 16
    header = ["Experiment", "Accuracy", "Macro-P", "Macro-R", "Macro-F1",
              "Params(M)", "FLOPs(M)", "Model(MB)", "FPS", "Lat(ms)", "PeakMem(MB)"]
    total_width = 22 + col_width * (len(header) - 1)

    print("\n\n" + "=" * total_width)
    print(f"{'🎯 FINAL ABLATION STUDY SUMMARY (5-Fold CV) 🎯':^{total_width}}")
    print("=" * total_width)
    print(f"{header[0]:<22}" + "".join(f"{h:>{col_width}}" for h in header[1:]))
    print("-" * total_width)

    for res in all_results:
        vals = [res['Accuracy'], res['Macro-Prec'], res['Macro-Recall'], res['Macro-F1'],
                res['Params(M)'], res['FLOPs(M)'], res['Model Size(MB)'],
                res['FPS'], res['Latency(ms)'], res['Peak Memory(MB)']]
        print(f"{res['Experiment']:<22}" + "".join(f"{v:>{col_width}}" for v in vals))
    print("=" * total_width)