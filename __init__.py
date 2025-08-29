# /ComfyUI/custom_nodes/MangaInpaintToolbox/__init__.py

from .nodes import (
    InteractivePanelCreator, MangaPanelDetector_Ultimate,
    CropPanelForInpaint_Advanced, ConditionalLatentScaler_Final,
    LatentTargetScaler_Pro, AssembleAndProgress,
    LayoutExtractor_Ultimate, LoadMangaFromOutput,
    FinalAssembler_Manga, PanelUpscalerForHires,
    ProgressiveUpscaleAssembler, # <--- 新ノードをインポート
)

WEB_DIRECTORY = "./js"
WEB_EXTENSIONS = { "MangaInpaintToolbox": "/extensions/MangaInpaintToolbox/main.js", }

NODE_CLASS_MAPPINGS = {
    "InteractivePanelCreator_Manga": InteractivePanelCreator,
    "MangaPanelDetector_Ultimate": MangaPanelDetector_Ultimate,
    "CropPanelForInpaint_Advanced": CropPanelForInpaint_Advanced,
    "ConditionalLatentScaler_Final": ConditionalLatentScaler_Final,
    "LatentTargetScaler_Pro_Manga": LatentTargetScaler_Pro,
    "AssembleAndProgress_Manga": AssembleAndProgress,
    "LayoutExtractor_Ultimate": LayoutExtractor_Ultimate,
    "LoadMangaFromOutput_Manga": LoadMangaFromOutput,
    "PanelUpscalerForHires_Manga": PanelUpscalerForHires,
    "FinalAssembler_Manga": FinalAssembler_Manga,
    "ProgressiveUpscaleAssembler_Manga": ProgressiveUpscaleAssembler, # <--- 新ノードを追加
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- Generation Workflow ---
    "InteractivePanelCreator_Manga": "① Create Panel Layout",
    "MangaPanelDetector_Ultimate": "① Detect Panels by Color",
    "CropPanelForInpaint_Advanced": "② Crop Panel (Shape Aware)",
    "LatentTargetScaler_Pro_Manga": "③ Latent Target Scaler (Pro)",
    "AssembleAndProgress_Manga": "④ Assemble Panel (Progressive)",
    "ConditionalLatentScaler_Final": "Legacy/Conditionally Scale Latent",
    
    # --- Upscale Workflow ---
    "LoadMangaFromOutput_Manga":   "U-① Load Manga from OUTPUT",
    "LayoutExtractor_Ultimate":    "U-② Extract Layout from Image",
    "PanelUpscalerForHires_Manga": "U-③ Panel Upscaler for Hires",
    "ProgressiveUpscaleAssembler_Manga": "U-④ Progressive Upscale Assembler", # <--- 新ノードの表示名
    "FinalAssembler_Manga":        "Legacy/Assemble Upscaled Panel (Stateless)", # <--- 旧ノード
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Toolbox (with Progressive Upscaler) ###")