import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# -------- Configuration --------
IMAGE_PATH = "test_image2.jpg"  # Replace with your image path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_COLORS = np.array([
    [0, 0, 0],        # 0=background
    [128, 0, 0],      # 1=aeroplane
    [0, 128, 0],      # 2=bicycle
    [128, 128, 0],    # 3=bird
    [0, 0, 128],      # 4=boat
    [128, 0, 128],    # 5=bottle
    [0, 128, 128],    # 6=bus
    [128, 128, 128],  # 7=car
    [64, 0, 0],       # 8=cat
    [192, 0, 0],      # 9=chair
    [64, 128, 0],     # 10=cow
    [192, 128, 0],    # 11=diningtable
    [64, 0, 128],     # 12=dog
    [192, 0, 128],    # 13=horse
    [64, 128, 128],   # 14=motorbike
    [192, 128, 128],  # 15=person
    [0, 64, 0],       # 16=pottedplant
    [128, 64, 0],     # 17=sheep
    [0, 192, 0],      # 18=sofa
    [128, 192, 0],    # 19=train
    [0, 64, 128]      # 20=tv/monitor
])

# -------- Load Model --------
model = models.deeplabv3_resnet50(pretrained=True).to(DEVICE)
model.eval()

# -------- Load and Preprocess Image --------
def preprocess_image(path):
    image = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return image, transform(image).unsqueeze(0)

raw_img, input_tensor = preprocess_image(IMAGE_PATH)
input_tensor = input_tensor.to(DEVICE)

# -------- Inference --------
with torch.no_grad():
    output = model(input_tensor)["out"][0]
    pred = output.argmax(0).cpu().numpy()

# -------- Decode Segmentation --------
def decode_segmentation(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(CLASS_COLORS)):
        color_mask[mask == class_id] = CLASS_COLORS[class_id]
    return color_mask

seg_mask = decode_segmentation(pred)

# -------- Visualization --------
def show_segmentation(original, mask, save_path="result.png"):
    original = original.resize((256, 256))
    blended = cv2.addWeighted(np.array(original), 0.5, mask, 0.5, 0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segmentation Mask")
    plt.imshow(mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(blended)
    plt.axis("off")

    plt.tight_layout()

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

show_segmentation(raw_img, seg_mask)
