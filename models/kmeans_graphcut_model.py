import numpy as np
import matplotlib.pyplot as plt
import maxflow
import os
import time
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
    # trả về (số vùng, ảnh nhãn, dict {nhãn: {'area', 'y0','y1','x0','x1'}})
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
                y0 = y1 = i
                x0 = x1 = j
                area = 0
                while q:
                    y, x = q.popleft()
                    area += 1
                    y0 = min(y0, y); y1 = max(y1, y)
                    x0 = min(x0, x); x1 = max(x1, x)
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and visited[ny, nx]:
                            visited[ny, nx] = 0
                            labels[ny, nx] = current_label
                            q.append((ny, nx))
                stats[current_label] = {'area': area, 'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1}
    return current_label, labels, stats

def khoi_tao_tam(X, k):
    n_samples = X.shape[0]
    centroids = np.zeros((k, 1))
    initial_idx = np.random.choice(n_samples)
    centroids[0] = X[initial_idx]
    for i in range(1, k):
        distances = np.sqrt((X - centroids[:i].T) ** 2)
        min_distances = np.min(distances, axis=1)
        squared_distances = min_distances ** 2
        total = np.sum(squared_distances)
        if total == 0:
            probabilities = np.ones(n_samples) / n_samples
        else:
            probabilities = squared_distances / total
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids[i] = X[next_idx]
    return centroids

def khoang_cach_euclid(X, centroids):
    return np.sqrt((X - centroids.T) ** 2)

def gan_nhan(distances):
    return np.argmin(distances, axis=1)

def cap_nhat_tam(X, labels, k):
    new_centroids = np.zeros((k, 1))
    for i in range(k):
        points_in_cluster = X[labels == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster)
        else:
            new_centroids[i] = X[np.random.choice(X.shape[0])]
    return new_centroids

def kmeans(X, k, max_iters=100, tolerance=1e-4):
    np.random.seed(42)
    centroids = khoi_tao_tam(X, k)
    for _ in range(max_iters):
        previous_centroids = centroids.copy()
        distances = khoang_cach_euclid(X, centroids)
        labels = gan_nhan(distances)
        centroids = cap_nhat_tam(X, labels, k)
        if np.all(np.abs(centroids - previous_centroids) < tolerance):
            break
    return centroids, labels

def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt   = gt_mask.flatten().astype(bool)
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)
    dsc = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return dsc, iou, sensitivity, specificity

def measure_tumor_size(pred_mask, pixel_spacing=0.26):
    num_labels, labels, stats = gan_nhan_lien_thong(pred_mask)
    if num_labels == 0:
        return None, 0, 0
    best_lbl = max(stats, key=lambda l: stats[l]['area'])
    s = stats[best_lbl]
    x, y = s['x0'], s['y0']
    w, h = s['x1'] - s['x0'] + 1, s['y1'] - s['y0'] + 1
    return (x, y, w, h), w * pixel_spacing, h * pixel_spacing

def get_brain_roi(image):
    binary = nguong_anh(image, 15, 255)
    kernel = tao_kernel_elip(11)
    closed = dong_hinh(binary, kernel, iterations=8)
    num_labels, labels, stats = gan_nhan_lien_thong(closed)
    if num_labels < 1:
        return np.zeros_like(image)
    best_lbl = max(stats, key=lambda l: stats[l]['area'])
    brain_roi = (labels == best_lbl).astype(np.uint8) * 255
    return brain_roi

def get_largest_component(mask, H, W, min_ratio=0.003, max_ratio=0.25):
    num_labels, labels, stats = gan_nhan_lien_thong(mask)
    best_area = 0
    best_component = None
    for lbl in range(1, num_labels + 1):
        s = stats[lbl]
        touches_border = (s['y0'] == 0 or s['y1'] == H - 1 or s['x0'] == 0 or s['x1'] == W - 1)
        if touches_border:
            continue
        area_ratio = s['area'] / (H * W)
        if area_ratio < min_ratio or area_ratio > max_ratio:
            continue
        if s['area'] > best_area:
            best_area = s['area']
            best_component = (labels == lbl).astype(np.uint8)
    return best_component

def graphcut_segment(image, fg_mask, bg_mask, fg_component):
    H, W = image.shape
    img_float = image.astype(np.float32) / 255.0
    # Giới hạn vùng GraphCut được phép hoạt động
    fg_area = np.sum(fg_mask > 0)
    fg_radius = int(np.sqrt(fg_area / np.pi))
    dilate_size = max(10, int(fg_radius * 1.2))
    if dilate_size % 2 == 0:
        dilate_size += 1
    kernel_roi = tao_kernel_elip(dilate_size)
    search_roi = gian_no(fg_mask, kernel_roi, iterations=2)
    g = maxflow.Graph[float](H * W, H * W * 8)
    nodes = g.add_nodes(H * W)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            intensity = img_float[i, j]
            # Pixel ngoài search_roi → ép về background
            if search_roi[i, j] == 0:
                g.add_tedge(idx, 0, 1e9)
                continue
            if fg_mask[i, j] > 0:
                source_w = 1e9
            else:
                source_w = intensity
            if bg_mask[i, j] > 0:
                sink_w = 1e9
            else:
                sink_w = 1.0 - intensity
            g.add_tedge(idx, source_w, sink_w)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if j + 1 < W:
                nidx = i * W + (j + 1)
                diff = abs(img_float[i, j] - img_float[i, j + 1])
                w = np.exp(-diff * 5)
                g.add_edge(idx, nidx, w, w)
            if i + 1 < H:
                nidx = (i + 1) * W + j
                diff = abs(img_float[i, j] - img_float[i + 1, j])
                w = np.exp(-diff * 5)
                g.add_edge(idx, nidx, w, w)
    g.maxflow()
    segments = g.get_grid_segments(nodes).reshape(H, W)
    gc_result = (~segments).astype(np.uint8)
    result = (gc_result | (fg_component > 0).astype(np.uint8))
    return result

def segment_one_image(image, k=5):
    H, W = image.shape
    brain_roi = get_brain_roi(image)
    brain_pixels_idx = np.where(brain_roi.flatten() > 0)[0]
    if len(brain_pixels_idx) == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        return empty, empty
    pixels_all = (image / 255.0).reshape(-1, 1)
    pixels_brain = pixels_all[brain_pixels_idx]
    centroids, labels_brain = kmeans(pixels_brain, k)
    labels_full = np.full(H * W, -1, dtype=int)
    labels_full[brain_pixels_idx] = labels_brain
    labels_img = labels_full.reshape(H, W)
    brightness_order = np.argsort([centroids[i][0] for i in range(k)])[::-1]
    total_pixels = H * W
    brain_roi_bin = (brain_roi // 255).astype(np.uint8)
    fg_component = None
    for cluster_idx in brightness_order:
        candidate = (labels_img == cluster_idx).astype(np.uint8) & brain_roi_bin
        area_ratio = np.sum(candidate) / total_pixels
        if area_ratio < 0.003 or area_ratio > 0.12:
            continue
        component = get_largest_component(candidate, H, W, min_ratio=0.003, max_ratio=0.12)
        if component is not None:
            fg_component = component
            break
    if fg_component is None:
        empty = np.zeros((H, W), dtype=np.uint8)
        return empty, empty
    kmeans_only_mask = fg_component.copy()
    kernel_close = tao_kernel_elip(15)
    fg_component_closed = dong_hinh(fg_component, kernel_close, iterations=3)
    fg_seed = fg_component_closed * 255
    bg_seed = 255 - brain_roi  # thay cv2.bitwise_not (brain_roi chỉ có giá trị 0/255)
    pred_mask = graphcut_segment(image, fg_seed, bg_seed, fg_component_closed)
    pred_mask = pred_mask & brain_roi_bin
    result = get_largest_component(pred_mask, H, W)
    if result is None:
        result = np.zeros((H, W), dtype=np.uint8)
    return kmeans_only_mask, result

def visualize_result(image, gt_mask, kmeans_mask, final_mask, dsc_km, iou_km, dsc_gc, iou_gc, idx=0):
    bbox, width_mm, height_mm = measure_tumor_size(final_mask)

    compare_overlay = np.stack([image] * 3, axis=-1).astype(np.uint8).copy()
    only_km = (kmeans_mask == 1) & (final_mask == 0)
    only_gc = (kmeans_mask == 0) & (final_mask == 1)
    both    = (kmeans_mask == 1) & (final_mask == 1)
    compare_overlay[only_km] = [0, 255, 0]
    compare_overlay[only_gc] = [255, 0, 0]
    compare_overlay[both]    = [255, 255, 0]

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(f'So sánh KMeans vs KMeans+GraphCut - Ảnh #{idx}', fontsize=13, fontweight='bold')
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    axes[2].imshow(kmeans_mask, cmap='gray')
    axes[2].set_title(f'KMeans\nDSC={dsc_km:.4f} | IoU={iou_km:.4f}')
    axes[2].axis('off')

    axes[3].imshow(final_mask, cmap='gray')
    if bbox is not None:
        x, y, w, h = bbox
        axes[3].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='white', linewidth=1))
        axes[3].text(x, max(y - 8, 0), f'{width_mm:.1f}mm x {height_mm:.1f}mm', color='white', fontsize=7)
    axes[3].set_title(f'KMeans+GraphCut\nDSC={dsc_gc:.4f} | IoU={iou_gc:.4f}')
    axes[3].axis('off')

    axes[4].imshow(compare_overlay)
    axes[4].set_title('So sánh')
    axes[4].axis('off')

    final_overlay = np.stack([image] * 3, axis=-1).astype(np.uint8).copy()
    final_overlay[final_mask == 1] = [255, 0, 0]
    axes[5].imshow(final_overlay)
    axes[5].set_title('Overlay kết quả cuối\n(đỏ = dự đoán)')
    axes[5].axis('off')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/kmeans_graphcut_compare_{idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_kmeans_graphcut(images, masks, k=5, visualize_idx=0):
    km_dsc, km_iou, km_sens, km_spec = [], [], [], []
    gc_dsc, gc_iou, gc_sens, gc_spec = [], [], [], []
    t_start = time.time()
    for image, gt_mask in zip(images, masks):
        kmeans_mask, final_mask = segment_one_image(image, k=k)
        d, i, se, sp = compute_metrics(kmeans_mask, gt_mask)
        km_dsc.append(d); km_iou.append(i); km_sens.append(se); km_spec.append(sp)
        d, i, se, sp = compute_metrics(final_mask, gt_mask)
        gc_dsc.append(d); gc_iou.append(i); gc_sens.append(se); gc_spec.append(sp)
    t_end = time.time()
    t_total = t_end - t_start
    t_avg = t_total / len(images)
    kmeans_sample, final_sample = segment_one_image(images[visualize_idx], k=k)
    dsc_km_s, iou_km_s, _, _ = compute_metrics(kmeans_sample, masks[visualize_idx])
    dsc_gc_s, iou_gc_s, _, _ = compute_metrics(final_sample, masks[visualize_idx])
    visualize_result(images[visualize_idx], masks[visualize_idx], kmeans_sample, final_sample, dsc_km_s, iou_km_s, dsc_gc_s, iou_gc_s, idx=visualize_idx)
    print("\nChỉ số đánh giá KMeans:")
    print(f"DSC trung bình: {np.mean(km_dsc):.4f}")
    print(f"IoU trung bình: {np.mean(km_iou):.4f}")
    print(f"Sensitivity trung bình : {np.mean(km_sens):.4f}")
    print(f"Specificity trung bình : {np.mean(km_spec):.4f}")
    print("\nChỉ số đánh giá KMeans + GraphCut:")
    print(f"DSC trung bình: {np.mean(gc_dsc):.4f}")
    print(f"IoU trung bình: {np.mean(gc_iou):.4f}")
    print(f"Sensitivity trung bình : {np.mean(gc_sens):.4f}")
    print(f"Specificity trung bình : {np.mean(gc_spec):.4f}")
    print("\nChênh lệch:")
    print(f"ΔDSC: {np.mean(gc_dsc) - np.mean(km_dsc):+.4f}")
    print(f"ΔIoU: {np.mean(gc_iou) - np.mean(km_iou):+.4f}")
    print("\nThời gian chạy:")
    print(f"Tổng thời gian: {t_total:.2f} giây")
    print(f"Trung bình/ảnh: {t_avg:.3f} giây")