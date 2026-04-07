import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    vgg16, VGG16_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models with updated weights API ---
models_dict = {
    "VGG16": vgg16(weights=VGG16_Weights.DEFAULT).to(device).eval(),
    "ResNet50": resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval(),
    "DenseNet121": densenet121(weights=DenseNet121_Weights.DEFAULT).to(device).eval(),
    "MobileNetV2": mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device).eval(),
}

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# --- Grad-CAM function ---
def grad_cam(model, img_pil):
    target = {}

    # Hook last conv layer depending on model type
    if isinstance(model, type(vgg16(weights=VGG16_Weights.DEFAULT))):
        layer = model.features[-1]
    elif isinstance(model, type(resnet50(weights=ResNet50_Weights.DEFAULT))):
        layer = model.layer4[-1].conv3
    elif isinstance(model, type(densenet121(weights=DenseNet121_Weights.DEFAULT))):
        layer = model.features.denseblock4.denselayer16.conv2
    elif isinstance(model, type(mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT))):
        # Fix for torchvision >=0.13
        layer = model.features[-1][0]  # first conv in last block
    else:
        raise ValueError("Unsupported model type")

    def forward_hook(module, input, output):
        target['feat'] = output

    def backward_hook(module, grad_input, grad_output):
        target['grad'] = grad_output[0]

    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)

    # Preprocess
    x = preprocess(img_pil).unsqueeze(0).to(device)
    x.requires_grad = True

    # Forward
    logits = model(x)
    c = logits.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    logits[0, c].backward()

    # Grad-CAM
    A = target['feat'][0]
    dY = target['grad'][0]
    weights = dY.mean(dim=(1, 2))
    cam = (weights[:, None, None] * A).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False)
    return cam.squeeze().detach().cpu().numpy()


# --- Overlay function ---
def overlay_cam(img_pil, cam, alpha=0.5):
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    heatmap = plt.get_cmap('jet')(cam)[..., :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap
    return np.uint8(overlay * 255)


# --- Load image ---
img_path = r"C:\Users\kikit\OneDrive\Documents\UNIMAP PHD\A. TRAIN DATASET SPLIT\TESTING (10%)\MYC+\P2152-15 B192615_patch_88_78.png"
img = Image.open(img_path).convert("RGB")

# --- Generate overlays for all models ---
overlays = {}
for name, model in models_dict.items():
    cam = grad_cam(model, img)
    overlays[name] = overlay_cam(img, cam)

# --- Display each overlay sequentially ---
for name, overlay in overlays.items():
    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f"Grad-CAM Overlay: {name}")
    plt.show()   # this will pause until you close the figure