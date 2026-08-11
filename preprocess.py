# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import cv2
# import random

# def remove_skull(image):
#     _, binary = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
#     num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(closed)
#     if num_labels < 2:
#         return image
#     largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     brain_mask = (labeled == largest).astype(np.uint8) * 255
#     kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
#     brain_mask = cv2.erode(brain_mask, kernel_erode, iterations=2)
#     return cv2.bitwise_and(image, brain_mask)

# def preprocess_image(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = img.astype(np.float32)
#     #Áp dụng công thức Min-Max Scaling
#     igmin, igmax = img.min(), img.max()
#     if igmax - igmin > 1e-8:
#         img = ((img - igmin) / (igmax - igmin) * 255.0).astype(np.uint8)
#     img = remove_skull(img)
#     return img

# def preprocess_mask(mask_path):
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#     return mask

# def load_dataset(dataset_dir=r"C:\Users\tbgbo\Downloads\combined_dataset", num_samples=100, seed=42):
#     images_dir = os.path.join(dataset_dir, "images")
#     masks_dir  = os.path.join(dataset_dir, "masks")
#     all_files = sorted([
#         f for f in os.listdir(images_dir) if f.endswith(".png")
#     ])
#     #random.seed(seed)
#     selected = random.sample(all_files, min(num_samples, len(all_files)))
#     images, masks, names = [], [], []
#     skipped = 0
#     for fname in selected:
#         # img_path  = r"C:\Users\tbgbo\Downloads\TCGA_CS_4942_19970222_11.png"
#         # mask_path = r"C:\Users\tbgbo\Downloads\TCGA_CS_4942_19970222_11_mask.png"
#         img_path  = os.path.join(images_dir, fname)
#         mask_path = os.path.join(masks_dir,  fname)
#         if not os.path.exists(mask_path):
#             skipped += 1
#             continue
#         try:
#             img  = preprocess_image(img_path)
#             mask = preprocess_mask(mask_path)
#             images.append(img)
#             masks.append(mask)
#             names.append(fname)
#         except Exception as e:
#             print(f"  Loi {fname}: {e}")
#             skipped += 1
#     print(f"Da load {len(images)} anh (bo qua: {skipped})")
#     n_fig = sum(1 for n in names if n.startswith("figshare"))
#     n_lgg = sum(1 for n in names if n.startswith("lgg"))
#     print(f"  Figshare: {n_fig}  |  LGG: {n_lgg}")
#     return np.array(images), np.array(masks), names

# def visualize_samples(images, masks, names, n=3):
#     fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
#     indices = random.sample(range(len(images)), min(n, len(images)))
#     for row, idx in enumerate(indices):
#         axes[row, 0].imshow(images[idx], cmap='gray')
#         axes[row, 0].set_title(f'Anh sau xu ly\n{names[idx]}')
#         axes[row, 0].axis('off')

#         axes[row, 1].imshow(masks[idx], cmap='gray')
#         axes[row, 1].set_title('Ảnh nhãn')
#         axes[row, 1].axis('off')

#         overlay = cv2.cvtColor(images[idx], cv2.COLOR_GRAY2BGR)
#         overlay[masks[idx] > 0] = [0, 0, 200]
#         axes[row, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#         axes[row, 2].set_title('Chèn ảnh')
#         axes[row, 2].axis('off')
#     plt.tight_layout()
#     plt.show()

# def visualize_skull_stripping(images, names, n=3):
#     fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
#     indices = random.sample(range(len(images)), min(n, len(images)))
#     images_dir = os.path.join(r"C:\Users\tbgbo\Downloads\combined_dataset", "images")
#     for row, idx in enumerate(indices):
#         raw = cv2.imread(os.path.join(images_dir, names[idx]), cv2.IMREAD_GRAYSCALE)
#         stripped = images[idx]
#         diff = cv2.absdiff(raw, stripped)

#         axes[row, 0].imshow(raw, cmap='gray')
#         axes[row, 0].set_title(f'Anh goc\n{names[idx]}')
#         axes[row, 0].axis('off')

#         axes[row, 1].imshow(stripped, cmap='gray')
#         axes[row, 1].set_title('Sau xóa vỏ não')
#         axes[row, 1].axis('off')

#         axes[row, 2].imshow(diff, cmap='hot')
#         axes[row, 2].set_title('Phần bị xóa')
#         axes[row, 2].axis('off')
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     images, masks, names = load_dataset()
#     print(f"\nKích thước ảnh : {images.shape}")
#     print(f"Kích thước nhãn  : {masks.shape}")
#     #visualize_skull_stripping(images, names, n=2)
#     #visualize_samples(images, masks, names, n=2)

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from collections import deque

def tao_kernel_elip(size):
    if size % 2 == 0:
        size += 1
    r = size // 2
    kernel = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    kernel[(xx**2 + yy**2) <= r**2 + 1e-6] = 1
    return kernel

def gian_no(mask, kernel, iterations=1):
    m = (mask > 0).astype(np.uint8)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    offsets = [(i, j) for i in range(kh) for j in range(kw) if kernel[i, j]]
    for _ in range(iterations):
        padded = np.pad(m, ((ph, ph), (pw, pw)), constant_values=0)
        out = np.zeros_like(m)
        for i, j in offsets:
            out |= padded[i:i + m.shape[0], j:j + m.shape[1]]
        m = out
    return m

def co_lai(mask, kernel, iterations=1):
    m = (mask > 0).astype(np.uint8)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    offsets = [(i, j) for i in range(kh) for j in range(kw) if kernel[i, j]]
    for _ in range(iterations):
        padded = np.pad(m, ((ph, ph), (pw, pw)), constant_values=0)
        out = np.ones_like(m)
        for i, j in offsets:
            out &= padded[i:i + m.shape[0], j:j + m.shape[1]]
        m = out
    return m

def dong_hinh(mask, kernel, iterations=1):
    return co_lai(gian_no(mask, kernel, iterations), kernel, iterations)

def nguong_anh(image, thresh, maxval=255):
    return np.where(image > thresh, maxval, 0).astype(np.uint8)

def gan_nhan_lien_thong(binary):
    # trả về (số vùng, ảnh nhãn, dict {nhãn: diện tích})
    b = (binary > 0).astype(np.uint8)
    H, W = b.shape
    labels = np.zeros((H, W), dtype=np.int32)
    visited = b.copy()
    stats = {}
    current_label = 0
    for i in range(H):
        for j in range(W):
            if visited[i, j]:
                current_label += 1
                visited[i, j] = 0
                q = deque([(i, j)])
                labels[i, j] = current_label
                area = 0
                while q:
                    y, x = q.popleft()
                    area += 1
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and visited[ny, nx]:
                            visited[ny, nx] = 0
                            labels[ny, nx] = current_label
                            q.append((ny, nx))
                stats[current_label] = area
    return current_label, labels, stats

def remove_skull(image):
    binary = nguong_anh(image, 15, 255)
    kernel_close = tao_kernel_elip(11)
    closed = dong_hinh(binary, kernel_close, iterations=5)
    num_labels, labeled, stats = gan_nhan_lien_thong(closed)
    if num_labels < 1:
        return image
    largest_label = max(stats, key=stats.get)
    brain_mask = (labeled == largest_label).astype(np.uint8) * 255
    kernel_erode = tao_kernel_elip(20)
    brain_mask = co_lai(brain_mask, kernel_erode, iterations=2)
    result = np.where(brain_mask > 0, image, 0).astype(np.uint8)
    return result

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    igmin, igmax = img.min(), img.max()
    if igmax - igmin > 1e-8:
        img = ((img - igmin) / (igmax - igmin) * 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256))
    img = remove_skull(img)
    return img

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.shape != (256, 256):
        mask = cv2.resize(mask, (256, 256))
    mask = nguong_anh(mask, 127, 255)
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
            print(f"Loi {fname}: {e}")
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

        overlay = np.stack([images[idx]] * 3, axis=-1).astype(np.uint8).copy()
        overlay[masks[idx] > 0] = [200, 0, 0]
        axes[row, 2].imshow(overlay)
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
        diff = np.abs(raw.astype(np.int16) - stripped.astype(np.int16)).astype(np.uint8)

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