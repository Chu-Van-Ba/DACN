import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def remove_skull(image):
    _, binary = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels < 2:
        return image
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = (labeled == largest).astype(np.uint8) * 255
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    brain_mask = cv2.erode(brain_mask, kernel_erode, iterations=2)
    return cv2.bitwise_and(image, brain_mask)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    #Áp dụng công thức Min-Max Scaling
    igmin, igmax = img.min(), img.max()
    if igmax - igmin > 1e-8:
        img = ((img - igmin) / (igmax - igmin) * 255.0).astype(np.uint8)
    img = remove_skull(img)
    return img

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def load_dataset(dataset_dir=r"C:\Users\tbgbo\Downloads\combined_dataset", num_samples=100, seed=42):
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir  = os.path.join(dataset_dir, "masks")
    all_files = sorted([
        f for f in os.listdir(images_dir) if f.endswith(".png")
    ])
    #random.seed(seed)
    selected = random.sample(all_files, min(num_samples, len(all_files)))
    images, masks, names = [], [], []
    skipped = 0
    for fname in selected:
        # img_path  = r"C:\Users\tbgbo\Downloads\TCGA_CS_4942_19970222_11.png"
        # mask_path = r"C:\Users\tbgbo\Downloads\TCGA_CS_4942_19970222_11_mask.png"
        img_path  = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir,  fname)
        if not os.path.exists(mask_path):
            skipped += 1
            continue
        try:
            img  = preprocess_image(img_path)
            mask = preprocess_mask(mask_path)
            images.append(img)
            masks.append(mask)
            names.append(fname)
        except Exception as e:
            print(f"  Loi {fname}: {e}")
            skipped += 1
    print(f"Da load {len(images)} anh (bo qua: {skipped})")
    n_fig = sum(1 for n in names if n.startswith("figshare"))
    n_lgg = sum(1 for n in names if n.startswith("lgg"))
    print(f"  Figshare: {n_fig}  |  LGG: {n_lgg}")
    return np.array(images), np.array(masks), names

def visualize_samples(images, masks, names, n=3):
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    indices = random.sample(range(len(images)), min(n, len(images)))
    for row, idx in enumerate(indices):
        axes[row, 0].imshow(images[idx], cmap='gray')
        axes[row, 0].set_title(f'Anh sau xu ly\n{names[idx]}')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(masks[idx], cmap='gray')
        axes[row, 1].set_title('Ảnh nhãn')
        axes[row, 1].axis('off')

        overlay = cv2.cvtColor(images[idx], cv2.COLOR_GRAY2BGR)
        overlay[masks[idx] > 0] = [0, 0, 200]
        axes[row, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title('Chèn ảnh')
        axes[row, 2].axis('off')
    plt.tight_layout()
    plt.show()

def visualize_skull_stripping(images, names, n=3):
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    indices = random.sample(range(len(images)), min(n, len(images)))
    images_dir = os.path.join(r"C:\Users\tbgbo\Downloads\combined_dataset", "images")
    for row, idx in enumerate(indices):
        raw = cv2.imread(os.path.join(images_dir, names[idx]), cv2.IMREAD_GRAYSCALE)
        stripped = images[idx]
        diff = cv2.absdiff(raw, stripped)

        axes[row, 0].imshow(raw, cmap='gray')
        axes[row, 0].set_title(f'Anh goc\n{names[idx]}')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(stripped, cmap='gray')
        axes[row, 1].set_title('Sau xóa vỏ não')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(diff, cmap='hot')
        axes[row, 2].set_title('Phần bị xóa')
        axes[row, 2].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images, masks, names = load_dataset()
    print(f"\nKích thước ảnh : {images.shape}")
    print(f"Kích thước nhãn  : {masks.shape}")
    #visualize_skull_stripping(images, names, n=2)
    #visualize_samples(images, masks, names, n=2)