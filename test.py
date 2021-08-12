import torch
import torchvision.transforms as transforms
from PIL import Image
from model import VisionTransformer

image_size = 224
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
model = VisionTransformer(n_classes=1)
img = transform(Image.open("cat.jpg").convert("RGB")).unsqueeze(0).to(torch.float32)
print(model)
print(f"{img.shape=}")
out = model(img)
print(f"{out=}")
