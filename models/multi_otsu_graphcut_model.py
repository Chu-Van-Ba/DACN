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

def otsu_multi_threshold(image, mask, n_thresholds=2, n_bins=256):
    pixels = image[mask > 0]
    hist, _ = np.histogram(pixels, bins=n_bins, range=(0, 256))
    hist = hist.astype(np.float64) / hist.sum()  # chuẩn hóa thành xác suất
    bins = np.arange(n_bins)
    total_mean = np.sum(bins * hist)
    if n_thresholds == 1:
        # Otsu cơ bản — 1 ngưỡng, 2 vùng
        best_t = 0
        best_var = -1
        cum_w = np.cumsum(hist)
        cum_wm = np.cumsum(bins * hist)
        for t in range(1, n_bins - 1):
            w1 = cum_w[t]
            w2 = 1 - w1
            if w1 < 1e-8 or w2 < 1e-8:
                continue
            m1 = cum_wm[t] / w1
            m2 = (cum_wm[-1] - cum_wm[t]) / w2
            var_between = w1 * (m1 - total_mean) ** 2 + w2 * (m2 - total_mean) ** 2
            if var_between > best_var:
                best_var = var_between
                best_t = t
        return (best_t,)
    elif n_thresholds == 2:
        # Multi-Otsu — 2 ngưỡng, 3 vùng
        cum_w = np.cumsum(hist)
        cum_wm = np.cumsum(bins * hist)
        best_thresh = (1, 2)
        best_var = -1
        for t1 in range(1, n_bins - 2):
            w1 = cum_w[t1]
            if w1 < 1e-8:
                continue
            m1 = cum_wm[t1] / w1
            for t2 in range(t1 + 1, n_bins - 1):
                w2 = cum_w[t2] - cum_w[t1]
                w3 = 1 - cum_w[t2]
                if w2 < 1e-8 or w3 < 1e-8:
                    continue
                m2 = (cum_wm[t2] - cum_wm[t1]) / w2
                m3 = (cum_wm[-1] - cum_wm[t2]) / w3
                var_between = (w1 * (m1 - total_mean) ** 2 +
                               w2 * (m2 - total_mean) ** 2 +
                               w3 * (m3 - total_mean) ** 2)
                if var_between > best_var:
                    best_var = var_between
                    best_thresh = (t1, t2)
        return best_thresh
    else:
        raise ValueError("Chỉ hỗ trợ n_thresholds = 1 hoặc 2")

def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt   = gt_mask.flatten().astype(bool)
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)
    dsc         = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou         = tp / (tp + fp + fn + 1e-8)
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
    fg_area = np.sum(fg_mask > 0)
    fg_radius = int(np.sqrt(fg_area / np.pi))
    dilate_size = max(10, int(fg_radius * 0.8))
    if dilate_size % 2 == 0:
        dilate_size += 1
    kernel_roi = tao_kernel_elip(dilate_size)
    search_roi = gian_no(fg_mask, kernel_roi, iterations=2)
    g = maxflow.Graph[float](H * W, H * W * 4)
    nodes = g.add_nodes(H * W)
    # T-link
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            intensity = img_float[i, j]
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
    # N-link
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

def segment_one_image(image, n_thresholds=2):
    H, W = image.shape
    brain_roi = get_brain_roi(image)
    if np.sum(brain_roi) == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        return empty, empty
    thresholds = otsu_multi_threshold(image, brain_roi, n_thresholds=n_thresholds)
    # Với n_thresholds=2: thresholds = (t1, t2) → 3 vùng: tối / trung bình / sáng
    t_brightest = thresholds[-1]  # ngưỡng cao nhất - ranh giới vùng sáng nhất
    candidate = ((image > t_brightest) & (brain_roi > 0)).astype(np.uint8)
    fg_component = get_largest_component(candidate, H, W, min_ratio=0.003, max_ratio=0.12)
    if fg_component is None:
        empty = np.zeros((H, W), dtype=np.uint8)
        return empty, empty
    otsu_only_mask = fg_component.copy()
    kernel_close = tao_kernel_elip(15)
    fg_component_closed = dong_hinh(fg_component, kernel_close, iterations=3)
    fg_seed = fg_component_closed * 255
    bg_seed = 255 - brain_roi
    pred_mask = graphcut_segment(image, fg_seed, bg_seed, fg_component_closed)
    brain_roi_bin = (brain_roi // 255).astype(np.uint8)
    pred_mask = pred_mask & brain_roi_bin
    result = get_largest_component(pred_mask, H, W)
    if result is None:
        result = np.zeros((H, W), dtype=np.uint8)
    return otsu_only_mask, result

def visualize_result(image, gt_mask, otsu_mask, final_mask, dsc_otsu, iou_otsu, dsc_gc, iou_gc, idx=0):
    bbox, width_mm, height_mm = measure_tumor_size(final_mask)

    compare_overlay = np.stack([image] * 3, axis=-1).astype(np.uint8).copy()
    only_otsu = (otsu_mask == 1) & (final_mask == 0)
    only_gc   = (otsu_mask == 0) & (final_mask == 1)
    both      = (otsu_mask == 1) & (final_mask == 1)
    compare_overlay[only_otsu] = [0, 255, 0]
    compare_overlay[only_gc]   = [255, 0, 0]
    compare_overlay[both]      = [255, 255, 0]

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(f'So sánh Multi-Otsu vs Multi-Otsu+GraphCut - Ảnh #{idx}', fontsize=13, fontweight='bold')

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(otsu_mask, cmap='gray')
    axes[2].set_title(f'Otsu thuần\nDSC={dsc_otsu:.4f} | IoU={iou_otsu:.4f}')
    axes[2].axis('off')

    axes[3].imshow(final_mask, cmap='gray')
    if bbox is not None:
        x, y, w, h = bbox
        # thay cv2.rectangle + cv2.putText
        axes[3].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='white', linewidth=1))
        axes[3].text(x, max(y - 8, 0), f'{width_mm:.1f}mm x {height_mm:.1f}mm', color='white', fontsize=7)
    axes[3].set_title(f'Otsu+GraphCut\nDSC={dsc_gc:.4f} | IoU={iou_gc:.4f}')
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
    plt.savefig(f'results/otsu_graphcut_compare_{idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_otsu_graphcut(images, masks, n_thresholds=2, visualize_idx=0):
    otsu_dsc, otsu_iou, otsu_sens, otsu_spec = [], [], [], []
    gc_dsc, gc_iou, gc_sens, gc_spec = [], [], [], []
    t_start = time.time()
    for image, gt_mask in zip(images, masks):
        otsu_mask, final_mask = segment_one_image(image, n_thresholds=n_thresholds)
        d, i, se, sp = compute_metrics(otsu_mask, gt_mask)
        otsu_dsc.append(d); otsu_iou.append(i); otsu_sens.append(se); otsu_spec.append(sp)
        d, i, se, sp = compute_metrics(final_mask, gt_mask)
        gc_dsc.append(d); gc_iou.append(i); gc_sens.append(se); gc_spec.append(sp)
    t_end = time.time()
    t_total = t_end - t_start
    t_avg = t_total / len(images)
    otsu_sample, final_sample = segment_one_image(images[visualize_idx], n_thresholds=n_thresholds)
    dsc_otsu_s, iou_otsu_s, _, _ = compute_metrics(otsu_sample, masks[visualize_idx])
    dsc_gc_s, iou_gc_s, _, _ = compute_metrics(final_sample, masks[visualize_idx])
    visualize_result(images[visualize_idx], masks[visualize_idx], otsu_sample, final_sample, dsc_otsu_s, iou_otsu_s, dsc_gc_s, iou_gc_s, idx=visualize_idx)
    print("\nChỉ số đánh giá Multi-Otsu:")
    print(f"DSC trung bình: {np.mean(otsu_dsc):.4f}")
    print(f"IoU trung bình: {np.mean(otsu_iou):.4f}")
    print(f"Sensitivity trung bình: {np.mean(otsu_sens):.4f}")
    print(f"Specificity trung bình: {np.mean(otsu_spec):.4f}")
    print("\nChỉ số đánh giá Multi-Otsu + GraphCut:")
    print(f"DSC trung bình: {np.mean(gc_dsc):.4f}")
    print(f"IoU trung bình: {np.mean(gc_iou):.4f}")
    print(f"Sensitivity trung bình: {np.mean(gc_sens):.4f}")
    print(f"Specificity trung bình: {np.mean(gc_spec):.4f}")
    print("\nChênh lệch:")
    print(f"ΔDSC: {np.mean(gc_dsc) - np.mean(otsu_dsc):+.4f}")
    print(f"ΔIoU: {np.mean(gc_iou) - np.mean(otsu_iou):+.4f}")
    print("\nThời gian chạy:")
    print(f"Tổng thời gian: {t_total:.2f} giây")
    print(f"Trung bình/ảnh: {t_avg:.3f} giây")