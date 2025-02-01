import os
import matplotlib.pyplot as plt
import cv2
from collections import Counter

# Path to dataset (Update this to the correct folder)
dataset_path = "sign_language_dataset/train/Indian"

# Get class names (subfolder names)
classes = sorted(os.listdir(dataset_path))
print(f"Classes found: {classes}")

# Count images per class
class_counts = {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in classes}
print("\nNumber of images per class:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")

# Visualize sample images
def show_sample_images(dataset_path, classes, num_samples=5):
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(12, len(classes) * 2))
    for i, cls in enumerate(classes):
        class_path = os.path.join(dataset_path, cls)
        sample_images = os.listdir(class_path)[:num_samples]
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(cls)
    plt.tight_layout()
    plt.show()

show_sample_images(dataset_path, classes)

# Check image size consistency
image_sizes = []
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    for img_name in os.listdir(class_path)[:20]:  # Check first 20 images per class
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            image_sizes.append(img.shape[:2])

# Print unique image sizes
unique_sizes = Counter(image_sizes)
print("\nUnique image sizes found:")
for size, count in unique_sizes.items():
    print(f"Size {size}: {count} images")
