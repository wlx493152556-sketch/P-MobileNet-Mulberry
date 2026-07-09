import os
import io
import time  # 引入时间模块用于测速
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

torch.set_num_threads(1)

# ==========================================
# 1. Network Architecture (PE-MobileNet)
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
        k = 5  
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
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
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
    def __init__(self, num_classes=3, width_mult=0.25, expand_ratio=3, use_pconv=False, use_eca=False):
        super().__init__()
        settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
                    [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]

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
# 2. Model Initialization & Weights Loading
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate PE-MobileNet (t=3, use_pconv=True, use_eca=True)
model = ProposedMobileNetV2(num_classes=3, width_mult=0.25, expand_ratio=3, use_pconv=True, use_eca=True)

WEIGHT_PATH = "/openbayes/home/系统/best_Proposed_(t=3+P+E)_fold4.pth"

if os.path.exists(WEIGHT_PATH):
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print(f"Successfully loaded best model weights: {WEIGHT_PATH}")
else:
    print(f"⚠️ Warning: Weight file not found at {WEIGHT_PATH}. Using un-trained initial weights!")

model.to(DEVICE)
model.eval()

# ==========================================
# 3. Data Preprocessing (Aligned with Validation Strategy)
# ==========================================
CLASSES = ['Disease Free leaves', 'Leaf Rust', 'Leaf spot']

inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. Core Inference with Dynamic Speed Test
# ==========================================
def predict_mulberry(image):
    if image is None:
        return {cls: 0.0 for cls in CLASSES}, "0.00 ms"
    
    # Preprocessing
    img_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
    
    # Warm up & Inference with precise timing
    with torch.no_grad():
        # Optional GPU synchronization for precise timing if using CUDA
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
    # Calculate exact latency in milliseconds
    latency = (end_time - start_time) * 1000
    latency_str = f"⚡ Real-time Inference Latency: {latency:.2f} ms ({DEVICE.upper()})"
    
    # Return both the probabilities and the speed text
    prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    return prob_dict, latency_str

# ==========================================
# 5. Internationalized Gradio UI Layout
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), 
               title="PE-MobileNet Mulberry Disease Diagnosis",
               css="footer {display: none !important;}") as demo:
    gr.Markdown("**Mulberry Leaf Disease Diagnosis System**")
    gr.Markdown("powered by PE-MobileNet")
    
    with gr.Row():
        with gr.Column():
            # Supports both live camera capture and image upload
            input_img = gr.Image(type="pil", sources=["webcam", "upload"], label="Capture or Upload Mulberry Leaf")
            btn = gr.Button("Start Diagnosis", variant="primary")
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Diagnostic Confidence")
            # Dynamic performance tracking box
            speed_txt = gr.Textbox(label="Real-time Speed Monitor", placeholder="Latency will be displayed here...", interactive=False)
            
    btn.click(fn=predict_mulberry, inputs=input_img, outputs=[output_label, speed_txt])
    
    '''
    # Technical highlights section
    gr.Markdown("""
    ### 🔬 Technical Benchmarks (5-Fold CV Summary):
    * **Model Size**: ~0.90 MB | **Parameters**: 0.20M | **FLOPs**: 28.10M
    * **Target Edge Latency**: ~6.86 ms | **Runtime RAM**: ~772 MB
    ---
    *Designed for deployment on resource-constrained agricultural edge devices.*
    """)
    '''

# Launch the server on OpenBayes expected port
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)