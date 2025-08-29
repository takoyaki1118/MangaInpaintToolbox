# /ComfyUI/custom_nodes/MangaInpaintToolbox/nodes.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import os
import shutil
import re
import random
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args
from threading import Lock

# --- グローバル定数 ---
MAX_PANELS = 32
PRESET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "presets")
INDEX_FILE = os.path.join(folder_paths.get_output_directory(), "manga_index.json")
index_lock = Lock()

os.makedirs(PRESET_DIR, exist_ok=True)

# --- ヘルパー関数 ---
def tensor_to_pil(tensor, batch_index=0):
    return Image.fromarray(np.clip(255. * tensor[batch_index].cpu().numpy(), 0, 255).astype(np.uint8))

def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_preset_files():
    if not os.path.exists(PRESET_DIR):
        return []
    return [f for f in os.listdir(PRESET_DIR) if f.endswith('.json')]

# --------------------------------------------------------------------
# Node 1: InteractivePanelCreator
# --------------------------------------------------------------------
class InteractivePanelCreator:
    PRESET_FILES = ["(Manual Canvas)"] + get_preset_files()
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "preset": (s.PRESET_FILES, ), "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}), "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}), "frame_color_hex": ("STRING", {"default": "#FFFFFF"}), "background_color_hex": ("STRING", {"default": "#000000"}), "regions_json": ("STRING", {"multiline": True, "default": "[]", "widget": "hidden"}), "save_preset_as": ("STRING", {"default": ""}), } }
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("layout_image", "mask_batch", "panel_count", "regions_json")
    FUNCTION = "create_layout_and_masks"
    CATEGORY = "Manga Toolbox"
    def hex_to_bgr(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
    def create_layout_and_masks(self, preset, width, height, frame_color_hex, background_color_hex, regions_json, save_preset_as):
        if save_preset_as:
            filename = re.sub(r'[\\/*?:"<>|]', "", save_preset_as)
            if not filename.endswith('.json'): filename += '.json'
            save_path = os.path.join(PRESET_DIR, filename)
            try:
                parsed_json = json.loads(regions_json)
                with open(save_path, 'w', encoding='utf-8') as f: json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                print(f"### MangaInpaintToolbox: Saved preset to {filename} ###")
            except Exception as e: print(f"Error saving preset: {e}")
        final_regions_json = regions_json
        if preset != "(Manual Canvas)":
            preset_path = os.path.join(PRESET_DIR, preset)
            if os.path.exists(preset_path):
                print(f"### MangaInpaintToolbox: Loading preset: {preset} ###")
                with open(preset_path, 'r', encoding='utf-8') as f: final_regions_json = f.read()
            else: print(f"Warning: Preset file not found: {preset}. Falling back to manual canvas.")
        try: regions = json.loads(final_regions_json)
        except json.JSONDecodeError: regions = []
        bg_color, frame_color = self.hex_to_bgr(background_color_hex), self.hex_to_bgr(frame_color_hex)
        canvas_cv = np.full((height, width, 3), bg_color, dtype=np.uint8)
        mask_list = []
        if regions:
            for region in regions:
                single_mask_np = np.zeros((height, width), dtype=np.uint8)
                region_type = region.get("type", "rect")
                if region_type == "rect" and all(k in region for k in ['x', 'y', 'w', 'h']):
                    x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
                    if w > 0 and h > 0:
                        cv2.rectangle(canvas_cv, (x, y), (x + w, y + h), frame_color, -1)
                        cv2.rectangle(single_mask_np, (x, y), (x + w, y + h), 255, -1)
                elif region_type == "poly" and "points" in region and len(region["points"]) >= 3:
                    pts = np.array([[p['x'], p['y']] for p in region["points"]], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(canvas_cv, [pts], frame_color)
                    cv2.fillPoly(single_mask_np, [pts], 255)
                mask_list.append(torch.from_numpy(single_mask_np.astype(np.float32) / 255.0))
        image_tensor = torch.from_numpy(cv2.cvtColor(canvas_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
        if not mask_list:
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return (image_tensor, empty_mask, 0, final_regions_json)
        return (image_tensor, torch.stack(mask_list), len(mask_list), final_regions_json)

# --------------------------------------------------------------------
# Node 2: AssembleAndProgress
# --------------------------------------------------------------------
class AssembleAndProgress:
    output_dir = folder_paths.get_output_directory()
    temp_dir = os.path.join(output_dir, "manga_inpaint_temp")
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "mode": (["Chronological (制作モード)", "Overlay (修正モード)"],), "base_image": ("IMAGE",), "generated_panel": ("IMAGE",), "mask_batch": ("MASK",), "panel_index": ("INT", {"default": 1, "min": 1}), "panel_count": ("INT", {"default": 1, "min": 1}), "regions_json": ("STRING", {"multiline": True, "widget": "hidden"}), "load_from_index_chrono": ("INT", {"default": 0, "min": 0, "label": "Load From (制作モード用)"}), "filename_prefix": ("STRING", {"default": "Manga"}), "job_id": ("STRING", {"default": "default_job"}), }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "assemble"
    CATEGORY = "Manga Toolbox"
    def assemble(self, mode, base_image, generated_panel, mask_batch, panel_index, panel_count, regions_json, load_from_index_chrono, filename_prefix, job_id, prompt=None, extra_pnginfo=None):
        job_dir = os.path.join(self.temp_dir, job_id)
        history_dir = os.path.join(job_dir, "history")
        latest_dir = os.path.join(job_dir, "latest")
        os.makedirs(history_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)
        device = base_image.device
        canvas_tensor = None
        if mode == "Chronological (制作モード)":
            load_from_index = load_from_index_chrono
            if panel_index == 1 and load_from_index == 0:
                canvas_tensor = base_image[0].clone()
                if os.path.exists(job_dir): shutil.rmtree(job_dir)
                os.makedirs(history_dir, exist_ok=True)
                os.makedirs(latest_dir, exist_ok=True)
            else:
                load_index = load_from_index if load_from_index > 0 else panel_index - 1
                if load_index > 0:
                    load_path = os.path.join(history_dir, f"composite_panel_{load_index}.png")
                    if os.path.exists(load_path):
                        loaded_img = Image.open(load_path).convert("RGB")
                        canvas_tensor = pil_to_tensor(loaded_img)[0].to(device)
                    else:
                        print(f"Warning: History file not found: {load_path}. Falling back to base_image.")
                        canvas_tensor = base_image[0].clone()
                else: canvas_tensor = base_image[0].clone()
        else:
            latest_file_path = os.path.join(latest_dir, "latest_composite.png")
            if os.path.exists(latest_file_path):
                loaded_img = Image.open(latest_file_path).convert("RGB")
                canvas_tensor = pil_to_tensor(loaded_img)[0].to(device)
            else: canvas_tensor = base_image[0].clone()
        index = panel_index - 1
        if index < 0 or index >= mask_batch.shape[0]: return (canvas_tensor.unsqueeze(0),)
        image_to_paste, mask = generated_panel[0], mask_batch[index]
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (canvas_tensor.unsqueeze(0),)
        y1, y2, x1, x2 = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
        h, w = y2 - y1 + 1, x2 - x1 + 1
        if h <= 0 or w <= 0: return (canvas_tensor.unsqueeze(0),)
        resized_image = F.interpolate(image_to_paste.permute(2, 0, 1).unsqueeze(0), size=(h.item(), w.item()), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        target_region = canvas_tensor[y1:y2+1, x1:x2+1]
        sub_mask = mask[y1:y2+1, x1:x2+1].unsqueeze(-1)
        pasted_region = torch.where(sub_mask > 0.5, resized_image, target_region)
        canvas_tensor[y1:y2+1, x1:x2+1] = pasted_region
        final_image_tensor = canvas_tensor.unsqueeze(0)
        latest_save_path = os.path.join(latest_dir, "latest_composite.png")
        tensor_to_pil(final_image_tensor).save(latest_save_path)
        if mode == "Chronological (制作モード)":
            history_save_path = os.path.join(history_dir, f"composite_panel_{panel_index}.png")
            tensor_to_pil(final_image_tensor).save(history_save_path)
        if panel_index == panel_count:
            full_output_folder, filename, counter, subfolder, filename_prefix_out = folder_paths.get_save_image_path(filename_prefix, self.output_dir, final_image_tensor[0].shape[1], final_image_tensor[0].shape[0])
            results = list()
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))
            for image in final_image_tensor:
                img = tensor_to_pil(image.unsqueeze(0))
                compress_level = 4
                if hasattr(args, 'output_quality') and args.output_quality < 100:
                    compress_level = 4 if args.output_quality > 50 else 0
                image_file = f"{filename}_{counter:05}_.png"
                img.save(os.path.join(full_output_folder, image_file), pnginfo=metadata, compress_level=compress_level)
                results.append({"filename": image_file, "subfolder": subfolder, "type": "output"})
                counter += 1
            filename = results[0]['filename']
            print(f"### MangaInpaintToolbox: Final image saved as {filename} ###")
            try:
                layout_data = json.loads(regions_json)
                with index_lock:
                    if os.path.exists(INDEX_FILE):
                        with open(INDEX_FILE, 'r', encoding='utf-8') as f: index_data = json.load(f)
                    else: index_data = {}
                    index_data[filename] = layout_data
                    with open(INDEX_FILE, 'w', encoding='utf-8') as f: json.dump(index_data, f, indent=2, ensure_ascii=False)
                    print(f"### MangaInpaintToolbox: Layout data for {filename} saved to index. ###")
            except Exception as e: print(f"Error: Could not save layout data to index file. {e}")
        return (final_image_tensor,)

# --------------------------------------------------------------------
# Node 3: MangaPanelDetector_Ultimate
# --------------------------------------------------------------------
class MangaPanelDetector_Ultimate:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "image": ("IMAGE",), "frame_color_hex": ("STRING", {"default": "#FFFFFF"}), "color_tolerance": ("INT", {"default": 10}), "gap_closing_scale": ("INT", {"default": 5}), "final_line_thickness": ("INT", {"default": 5}), "sort_panels_by": (["top-to-bottom", "left-to-right", "largest-first"],), "min_area": ("INT", {"default": 5000}), }}
    RETURN_TYPES, FUNCTION, CATEGORY = ("MASK", "INT"), "detect_panels", "Manga Toolbox"
    RETURN_NAMES = ("mask_batch", "panel_count")
    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def detect_panels(self, image, frame_color_hex, color_tolerance, gap_closing_scale, final_line_thickness, sort_panels_by, min_area):
        base_img_cv2 = (image[0].cpu().numpy() * 255).astype(np.uint8); img_h, img_w = base_img_cv2.shape[:2]; frame_color = np.array(self.hex_to_rgb(frame_color_hex)); lower, upper = np.maximum(0, frame_color - color_tolerance), np.minimum(255, frame_color + color_tolerance); color_mask = cv2.inRange(base_img_cv2, lower, upper); closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((gap_closing_scale, gap_closing_scale), np.uint8)); final_frame_mask = cv2.dilate(closed_mask, np.ones((final_line_thickness, final_line_thickness), np.uint8), iterations=1); num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_frame_mask, 4, cv2.CV_32S); panels_meta = [{'label_index': i, 'box': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), 'area': stats[i, cv2.CC_STAT_AREA]} for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area];
        if not panels_meta: return (torch.zeros((1, img_h, img_w), device=image.device), 0)
        if sort_panels_by == "largest-first": panels_meta.sort(key=lambda item: item['area'], reverse=True)
        elif sort_panels_by == "top-to-bottom": panels_meta.sort(key=lambda item: (item['box'][1], item['box'][0]))
        else: panels_meta.sort(key=lambda item: (item['box'][0], item['box'][1]))
        mask_list = [torch.from_numpy((labels == item['label_index']).astype(np.float32)).to(image.device) for item in panels_meta[:MAX_PANELS]]; return (torch.stack(mask_list), len(mask_list))

# --------------------------------------------------------------------
# Node 4: CropPanelForInpaint_Advanced
# --------------------------------------------------------------------
class CropPanelForInpaint_Advanced:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "image": ("IMAGE",), "mask_batch": ("MASK",), "panel_index": ("INT", {"default": 1}), "fill_color_hex": ("STRING", {"default": "#FFFFFF"}), }}
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE", "MASK"), "crop", "Manga Toolbox"
    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def crop(self, image, mask_batch, panel_index, fill_color_hex):
        index = panel_index - 1;
        if index < 0 or index >= mask_batch.shape[0]: return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        image_tensor, mask = image[0], mask_batch[index]; coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        fill_color = torch.tensor([c / 255.0 for c in self.hex_to_rgb(fill_color_hex)], device=image.device, dtype=torch.float32); masked_img = torch.where(mask.unsqueeze(-1) > 0.5, image_tensor, fill_color); y1, y2 = coords[:, 0].min(), coords[:, 0].max(); x1, x2 = coords[:, 1].min(), coords[:, 1].max(); return (masked_img[y1:y2+1, x1:x2+1, :].unsqueeze(0), mask[y1:y2+1, x1:x2+1].unsqueeze(0))

# --------------------------------------------------------------------
# Node 5: ConditionalLatentScaler_Final
# --------------------------------------------------------------------
class ConditionalLatentScaler_Final:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "samples": ("LATENT",), "threshold_pixel_width": ("INT", {"default": 512}), "threshold_pixel_height": ("INT", {"default": 512}), "comparison_logic": (["AND (Both)", "OR (Either)"],), "scale_factor_if_small": ("FLOAT", {"default": 1.5}), "scale_factor_if_large": ("FLOAT", {"default": 1.0}), "upscale_method": (["bicubic", "bilinear", "nearest-exact"],), } }
    RETURN_TYPES, FUNCTION, CATEGORY = ("LATENT", "STRING"), "scale", "Manga Toolbox"
    RETURN_NAMES = ("scaled_samples", "info")
    def scale(self, samples, threshold_pixel_width, threshold_pixel_height, comparison_logic, scale_factor_if_small, scale_factor_if_large, upscale_method):
        s, latent = samples.copy(), samples["samples"];
        if latent.shape[0] == 0: return (s, "Empty latent input.")
        current_pixel_w, current_pixel_h = latent.shape[3] * 8, latent.shape[2] * 8; is_small = (current_pixel_w < threshold_pixel_width and current_pixel_h < threshold_pixel_height) if comparison_logic == "AND (Both)" else (current_pixel_w < threshold_pixel_width or current_pixel_h < threshold_pixel_height); scale_factor = scale_factor_if_small if is_small else scale_factor_if_large;
        if scale_factor == 1.0: return (s, f"Not scaled. Current: {current_pixel_w}x{current_pixel_h}")
        new_h, new_w = int(latent.shape[2] * scale_factor), int(latent.shape[3] * scale_factor); s["samples"] = F.interpolate(latent, size=(new_h, new_w), mode=upscale_method); return (s, f"Scaled by x{scale_factor:.2f}. From {current_pixel_w}x{current_pixel_h} -> {new_w*8}x{new_h*8}")

# --------------------------------------------------------------------
# Node 6: LayoutExtractor_Ultimate
# --------------------------------------------------------------------
class LayoutExtractor_Ultimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "image": ("IMAGE",), "image_path": ("STRING", {"default": "ComfyUI_00001_.png"}), "frame_color_hex": ("STRING", {"default": "#FFFFFF"}), "color_tolerance": ("INT", {"default": 10}), "gap_closing_scale": ("INT", {"default": 5}), "final_line_thickness": ("INT", {"default": 5}), "sort_panels_by": (["top-to-bottom", "left-to-right", "largest-first"],), "min_area": ("INT", {"default": 5000}), }}
    RETURN_TYPES = ("MASK", "INT", "IMAGE")
    RETURN_NAMES = ("mask_batch", "panel_count", "original_image")
    FUNCTION = "extract_layout"
    CATEGORY = "Manga Toolbox"
    def hex_to_rgb(self, h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def extract_layout(self, image, image_path, frame_color_hex, color_tolerance, gap_closing_scale, final_line_thickness, sort_panels_by, min_area):
        filename = os.path.basename(image_path)
        img_h, img_w = image.shape[1], image.shape[2]
        layout_from_index = None
        if os.path.exists(INDEX_FILE):
            with index_lock:
                with open(INDEX_FILE, 'r', encoding='utf-8') as f: index_data = json.load(f)
            if filename in index_data:
                layout_from_index = index_data[filename]
                print(f"### LayoutExtractor: Found layout for '{filename}' in index. Using exact data. ###")
        if layout_from_index:
            mask_list = []
            for region in layout_from_index:
                single_mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
                region_type = region.get("type", "rect")
                if region_type == "rect" and all(k in region for k in ['x', 'y', 'w', 'h']):
                    x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
                    if w > 0 and h > 0: cv2.rectangle(single_mask_np, (x, y), (x + w, y + h), 255, -1)
                elif region_type == "poly" and "points" in region and len(region["points"]) >= 3:
                    pts = np.array([[p['x'], p['y']] for p in region["points"]], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(single_mask_np, [pts], 255)
                mask_list.append(torch.from_numpy(single_mask_np.astype(np.float32) / 255.0))
            if not mask_list: return (torch.zeros((1, img_h, img_w)), 0, image)
            return (torch.stack(mask_list).to(image.device), len(mask_list), image)
        print(f"### LayoutExtractor: Layout for '{filename}' not in index. Falling back to color detection. ###")
        base_img_cv2 = (image[0].cpu().numpy() * 255).astype(np.uint8)
        frame_color = np.array(self.hex_to_rgb(frame_color_hex))
        lower, upper = np.maximum(0, frame_color - color_tolerance), np.minimum(255, frame_color + color_tolerance)
        color_mask = cv2.inRange(base_img_cv2, lower, upper)
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((gap_closing_scale, gap_closing_scale), np.uint8))
        final_frame_mask = cv2.dilate(closed_mask, np.ones((final_line_thickness, final_line_thickness), np.uint8), iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_frame_mask, 4, cv2.CV_32S)
        panels_meta = [{'label_index': i, 'box': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), 'area': stats[i, cv2.CC_STAT_AREA]} for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
        if not panels_meta: return (torch.zeros((1, img_h, img_w), device=image.device), 0, image)
        if sort_panels_by == "largest-first": panels_meta.sort(key=lambda item: item['area'], reverse=True)
        elif sort_panels_by == "top-to-bottom": panels_meta.sort(key=lambda item: (item['box'][1], item['box'][0]))
        else: panels_meta.sort(key=lambda item: (item['box'][0], item['box'][1]))
        mask_list = [torch.from_numpy((labels == item['label_index']).astype(np.float32)) for item in panels_meta[:MAX_PANELS]]
        return (torch.stack(mask_list).to(image.device), len(mask_list), image)

# --------------------------------------------------------------------
# ★ Node 7: LoadMangaFromOutput (outputフォルダをスキャンする新ローダー) ★
# --------------------------------------------------------------------
class LoadMangaFromOutput:
    @classmethod
    def INPUT_TYPES(s):
        output_dir = folder_paths.get_output_directory()
        image_paths = []
        # outputフォルダ内のサブフォルダも含めて再帰的に検索
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # outputディレクトリからの相対パスを作成
                    relative_path = os.path.relpath(os.path.join(root, file), output_dir)
                    image_paths.append(relative_path)
        
        # 新しいものが上に来るように逆順ソート
        image_paths.sort(reverse=True)
        
        return {"required": {"image": (image_paths,)}}

    CATEGORY = "Manga Toolbox/Upscale Utils"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_path")
    FUNCTION = "load_image"

    def load_image(self, image):
        # image引数にはoutputフォルダからの相対パスが入っている
        image_path = os.path.join(folder_paths.get_output_directory(), image)
        
        i = Image.open(image_path)
        i = i.convert("RGB")
        image_tensor = pil_to_tensor(i)
        
        # 重要なのは、LayoutExtractorに渡すために「相対パス」を返すこと
        return (image_tensor, image)

# --------------------------------------------------------------------
# ★ Node 8: FinalAssembler_Manga (panel_index を受け取るように修正) ★
# --------------------------------------------------------------------
class FinalAssembler_Manga:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "base_image": ("IMAGE",),
                    "upscaled_panel": ("IMAGE",), # 単一のパネルを受け取るように変更
                    "mask_batch": ("MASK",),
                    "panel_index": ("INT", {"default": 1, "min": 1}), # panel_index を追加
                }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "assemble_final"
    CATEGORY = "Manga Toolbox/Upscale Utils"

    def assemble_final(self, base_image, upscaled_panel, mask_batch, panel_index):
        # base_imageはバッチかもしれないので、最初の1枚をキャンバスとして使用
        canvas = base_image[0].clone()
        
        # upscaled_panelもバッチかもしれないので、最初の1枚を使用
        panel_to_paste = upscaled_panel[0]

        index = panel_index - 1 # 0-based index

        if index < 0 or index >= len(mask_batch):
            print(f"Error: panel_index ({panel_index}) is out of bounds for mask_batch with size {len(mask_batch)}.")
            return (base_image,) # エラー時は元画像をそのまま返す

        mask = mask_batch[index]

        coords = torch.nonzero(mask, as_tuple=False)
        if coords.shape[0] == 0: return (canvas.unsqueeze(0),)

        y1, y2, x1, x2 = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
        h, w = y2 - y1 + 1, x2 - x1 + 1
        if h <= 0 or w <= 0: return (canvas.unsqueeze(0),)
        
        # panel_to_paste はすでに torch.Tensor なので permute と unsqueeze で次元を合わせる
        resized_image = F.interpolate(panel_to_paste.permute(2, 0, 1).unsqueeze(0), size=(h.item(), w.item()), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

        target_region = canvas[y1:y2+1, x1:x2+1]
        sub_mask = mask[y1:y2+1, x1:x2+1].unsqueeze(-1)
        pasted_region = torch.where(sub_mask > 0.5, resized_image, target_region)
        canvas[y1:y2+1, x1:x2+1] = pasted_region
            
        return (canvas.unsqueeze(0),)
