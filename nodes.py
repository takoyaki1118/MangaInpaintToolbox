# /ComfyUI/custom_nodes/MangaInpaintToolbox/nodes.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import os
import shutil
from PIL import Image

# --- グローバル定数 ---
MAX_PANELS = 32

# --- ヘルパー関数 ---
def tensor_to_pil(tensor, batch_index=0):
    return Image.fromarray(np.clip(255. * tensor[batch_index].cpu().numpy(), 0, 255).astype(np.uint8))

def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --------------------------------------------------------------------
# ★ Node 1: InteractivePanelCreator (多角形対応版) ★
# --------------------------------------------------------------------
class InteractivePanelCreator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "frame_color_hex": ("STRING", {"default": "#000000"}), # 枠線は黒
                "background_color_hex": ("STRING", {"default": "#FFFFFF"}), # 背景は白
                "regions_json": ("STRING", {"multiline": True, "default": "[]", "widget": "hidden"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_layout_image"
    CATEGORY = "Manga Toolbox"

    def hex_to_bgr(self, h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0)) # BGR

    def create_layout_image(self, width, height, frame_color_hex, background_color_hex, regions_json):
        try:
            regions = json.loads(regions_json)
        except json.JSONDecodeError:
            regions = []

        bg_color, frame_color = self.hex_to_bgr(background_color_hex), self.hex_to_bgr(frame_color_hex)
        canvas_cv = np.full((height, width, 3), bg_color, dtype=np.uint8)

        if regions:
            # 領域を逆順に描画して、リストの若い番号が手前に来るようにする
            for region in reversed(regions):
                region_type = region.get("type", "rect")
                if region_type == "rect" and all(k in region for k in ['x', 'y', 'w', 'h']):
                    x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
                    if w > 0 and h > 0:
                        cv2.rectangle(canvas_cv, (x, y), (x + w, y + h), frame_color, -1)
                elif region_type == "poly" and "points" in region and len(region["points"]) >= 3:
                    pts = np.array([[p['x'], p['y']] for p in region["points"]], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(canvas_cv, [pts], frame_color)

        canvas_rgb = cv2.cvtColor(canvas_cv, cv2.COLOR_BGR2RGB)
        return (torch.from_numpy(canvas_rgb.astype(np.float32) / 255.0).unsqueeze(0),)

# --------------------------------------------------------------------
# ★ Node 2: AssembleAndProgress (やり直し対応版) ★
# --------------------------------------------------------------------
class AssembleAndProgress:
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "output")
    temp_dir = os.path.join(output_dir, "manga_inpaint_temp")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "generated_panel": ("IMAGE",),
                "mask_batch": ("MASK",),
                "panel_index": ("INT", {"default": 1, "min": 1}),
                "load_from_index": ("INT", {"default": 0, "min": 0}),
                "job_id": ("STRING", {"default": "default_job"}),
            }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "assemble", "Manga Toolbox"

    def assemble(self, base_image, generated_panel, mask_batch, panel_index, load_from_index, job_id):
        job_dir = os.path.join(self.temp_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        canvas_tensor = None
        device = base_image.device

        if load_from_index == 0:
            canvas_tensor = base_image[0].clone()
            if panel_index == 1: # 最初のステップなら古いデータを消去
                shutil.rmtree(job_dir)
                os.makedirs(job_dir, exist_ok=True)
        else:
            load_path = os.path.join(job_dir, f"composite_panel_{load_from_index}.png")
            if os.path.exists(load_path):
                print(f"Loading previous state: {load_path}")
                loaded_img = Image.open(load_path).convert("RGB")
                canvas_tensor = pil_to_tensor(loaded_img)[0].to(device)
            else:
                print(f"Warning: State file not found: {load_path}. Falling back to base_image.")
                canvas_tensor = base_image[0].clone()
        
        index = panel_index - 1
        if index < 0 or index >= mask_batch.shape[0]: return (canvas_tensor.unsqueeze(0),)
        
        image_to_paste, mask = generated_panel[0], mask_batch[index]
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (canvas_tensor.unsqueeze(0),)

        y1, y2 = coords[:, 0].min(), coords[:, 0].max()
        x1, x2 = coords[:, 1].min(), coords[:, 1].max()
        h, w = y2 - y1 + 1, x2 - x1 + 1

        if h <= 0 or w <= 0: return (canvas_tensor.unsqueeze(0),)

        resized_image = F.interpolate(image_to_paste.permute(2, 0, 1).unsqueeze(0), size=(h.item(), w.item()), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        target_region = canvas_tensor[y1:y2+1, x1:x2+1]
        sub_mask = mask[y1:y2+1, x1:x2+1].unsqueeze(-1)
        pasted_region = torch.where(sub_mask > 0.5, resized_image, target_region)
        canvas_tensor[y1:y2+1, x1:x2+1] = pasted_region
        
        save_path = os.path.join(job_dir, f"composite_panel_{panel_index}.png")
        tensor_to_pil(canvas_tensor.unsqueeze(0)).save(save_path)
        print(f"Saved current state to: {save_path}")

        return (canvas_tensor.unsqueeze(0),)

# --------------------------------------------------------------------
# Node 3: MangaPanelDetector_Ultimate
# --------------------------------------------------------------------
class MangaPanelDetector_Ultimate:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",), "frame_color_hex": ("STRING", {"default": "#000000"}), "color_tolerance": ("INT", {"default": 10}), "gap_closing_scale": ("INT", {"default": 5}), "final_line_thickness": ("INT", {"default": 5}), "sort_panels_by": (["top-to-bottom", "left-to-right", "largest-first"],), "min_area": ("INT", {"default": 5000}), }}
    RETURN_TYPES, FUNCTION, CATEGORY = ("MASK", "INT"), "detect_panels", "Manga Toolbox"
    RETURN_NAMES = ("mask_batch", "panel_count")
    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def detect_panels(self, image, frame_color_hex, color_tolerance, gap_closing_scale, final_line_thickness, sort_panels_by, min_area):
        base_img_cv2 = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_h, img_w = base_img_cv2.shape[:2]
        frame_color = np.array(self.hex_to_rgb(frame_color_hex))
        lower, upper = np.maximum(0, frame_color - color_tolerance), np.minimum(255, frame_color + color_tolerance)
        color_mask = cv2.inRange(base_img_cv2, lower, upper)
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((gap_closing_scale, gap_closing_scale), np.uint8))
        final_frame_mask = cv2.dilate(closed_mask, np.ones((final_line_thickness, final_line_thickness), np.uint8), iterations=1)
        inverted_mask = cv2.bitwise_not(final_frame_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, 4, cv2.CV_32S)
        panels_meta = [{'label_index': i, 'box': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), 'area': stats[i, cv2.CC_STAT_AREA]} for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
        if not panels_meta: return (torch.zeros((1, img_h, img_w), device=image.device), 0)
        if sort_panels_by == "largest-first": panels_meta.sort(key=lambda item: item['area'], reverse=True)
        elif sort_panels_by == "top-to-bottom": panels_meta.sort(key=lambda item: (item['box'][1], item['box'][0]))
        else: panels_meta.sort(key=lambda item: (item['box'][0], item['box'][1]))
        mask_list = [torch.from_numpy((labels == item['label_index']).astype(np.float32)).to(image.device) for item in panels_meta[:MAX_PANELS]]
        return (torch.stack(mask_list), len(mask_list))

# --------------------------------------------------------------------
# Node 4: CropPanelForInpaint_Advanced
# --------------------------------------------------------------------
class CropPanelForInpaint_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",), "mask_batch": ("MASK",), "panel_index": ("INT", {"default": 1}), "fill_color_hex": ("STRING", {"default": "#FFFFFF"}), }}
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE", "MASK"), "crop", "Manga Toolbox"
    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def crop(self, image, mask_batch, panel_index, fill_color_hex):
        index = panel_index - 1
        if index < 0 or index >= mask_batch.shape[0]: return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        image_tensor, mask = image[0], mask_batch[index]
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        fill_color = torch.tensor([c / 255.0 for c in self.hex_to_rgb(fill_color_hex)], device=image.device, dtype=torch.float32)
        masked_img = torch.where(mask.unsqueeze(-1) > 0.5, image_tensor, fill_color)
        y1, y2 = coords[:, 0].min(), coords[:, 0].max()
        x1, x2 = coords[:, 1].min(), coords[:, 1].max()
        return (masked_img[y1:y2+1, x1:x2+1, :].unsqueeze(0), mask[y1:y2+1, x1:x2+1].unsqueeze(0))

# --------------------------------------------------------------------
# Node 5: ConditionalLatentScaler_Final
# --------------------------------------------------------------------
class ConditionalLatentScaler_Final:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "samples": ("LATENT",), "threshold_pixel_width": ("INT", {"default": 512}), "threshold_pixel_height": ("INT", {"default": 512}), "comparison_logic": (["AND (Both)", "OR (Either)"],), "scale_factor_if_small": ("FLOAT", {"default": 1.5}), "scale_factor_if_large": ("FLOAT", {"default": 1.0}), "upscale_method": (["bicubic", "bilinear", "nearest-exact"],), }
        }
    RETURN_TYPES, FUNCTION, CATEGORY = ("LATENT", "STRING"), "scale", "Manga Toolbox"
    RETURN_NAMES = ("scaled_samples", "info")
    def scale(self, samples, threshold_pixel_width, threshold_pixel_height, comparison_logic, scale_factor_if_small, scale_factor_if_large, upscale_method):
        s, latent = samples.copy(), samples["samples"]
        if latent.shape[0] == 0: return (s, "Empty latent input.")
        
        current_pixel_w, current_pixel_h = latent.shape[3] * 8, latent.shape[2] * 8
        is_small = (current_pixel_w < threshold_pixel_width and current_pixel_h < threshold_pixel_height) if comparison_logic == "AND (Both)" else (current_pixel_w < threshold_pixel_width or current_pixel_h < threshold_pixel_height)
        scale_factor = scale_factor_if_small if is_small else scale_factor_if_large
        
        if scale_factor == 1.0: return (s, f"Not scaled. Current: {current_pixel_w}x{current_pixel_h}")

        new_h, new_w = int(latent.shape[2] * scale_factor), int(latent.shape[3] * scale_factor)
        s["samples"] = F.interpolate(latent, size=(new_h, new_w), mode=upscale_method)
        return (s, f"Scaled by x{scale_factor:.2f}. From {current_pixel_w}x{current_pixel_h} -> {new_w*8}x{new_h*8}")