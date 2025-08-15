import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class RegionAttributes:
    bbox: tuple
    area_px: int
    rel_area: float
    mean_intensity: float
    edge_density: float
    circularity: float
    elongation: float
    centroid_xy: tuple
    loc_label: str

def _compute_loc_label(cx, cy, W, H):
    horiz = "left" if cx < W/3 else ("center" if cx < 2*W/3 else "right")
    vert = "upper" if cy < H/3 else ("mid" if cy < 2*H/3 else "lower")
    return f"{vert}-{horiz}"

def _safe_crop(img, x, y, w, h):
    H, W = img.shape[:2]
    x2, y2 = min(x+w, W), min(y+h, H)
    return img[max(y,0):y2, max(x,0):x2]

def describe_region(orig_img_gray, heatmap_01, region_bbox):
    x, y, w, h = region_bbox
    H, W = heatmap_01.shape[:2]
    patch_hm = _safe_crop(heatmap_01, x, y, w, h)
    patch_img = _safe_crop(orig_img_gray, x, y, w, h)

    mean_intensity = float(np.clip(patch_hm.mean(), 0, 1))
    edges = cv2.Canny(patch_img, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    area_px = int(w * h)
    rel_area = float(area_px / (W * H))

    th = (patch_hm > 0.5).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        circularity = 0.0
        elongation = 0.0
    else:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True) + 1e-6
        circularity = float(4 * np.pi * area / (peri * peri))
        x2,y2,w2,h2 = cv2.boundingRect(c)
        elongation = float(max(w2, h2) / (min(w2, h2) + 1e-6))

    cx = x + w/2
    cy = y + h/2
    loc_label = _compute_loc_label(cx, cy, W, H)

    return RegionAttributes(
        bbox=(int(x), int(y), int(w), int(h)),
        area_px=area_px,
        rel_area=rel_area,
        mean_intensity=mean_intensity,
        edge_density=edge_density,
        circularity=circularity,
        elongation=elongation,
        centroid_xy=(float(cx), float(cy)),
        loc_label=loc_label
    )
