import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load input image
image = Image.open("input_image.jpg").convert("RGB")

# Transform image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_tensor = transform(image)

# Simulated cGAN output (placeholder)
output_tensor = input_tensor

# Display images
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Input Image")

plt.subplot(1,2,2)
plt.imshow(output_tensor.permute(1,2,0))
plt.title("Translated Image")

plt.show()