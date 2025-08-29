# /ComfyUI/custom_nodes/MangaInpaintToolbox/__init__.py

from .nodes import (
    InteractivePanelCreator,
    MangaPanelDetector_Ultimate,
    CropPanelForInpaint_Advanced,
    ConditionalLatentScaler_Final,
    AssembleAndProgress,
    LayoutExtractor_Ultimate,
    LoadMangaFromOutput,      # <--- 新しいローダーをインポート
    FinalAssembler_Manga,
)

WEB_DIRECTORY = "./js"
WEB_EXTENSIONS = { "MangaInpaintToolbox": "/extensions/MangaInpaintToolbox/main.js", }

NODE_CLASS_MAPPINGS = {
    "InteractivePanelCreator_Manga": InteractivePanelCreator,
    "MangaPanelDetector_Ultimate": MangaPanelDetector_Ultimate,
    "CropPanelForInpaint_Advanced": CropPanelForInpaint_Advanced,
    "ConditionalLatentScaler_Final": ConditionalLatentScaler_Final,
    "AssembleAndProgress_Manga": AssembleAndProgress,
    "LayoutExtractor_Ultimate": LayoutExtractor_Ultimate,
    "LoadMangaFromOutput_Manga": LoadMangaFromOutput,         # <--- 新しいローダーを追加
    "FinalAssembler_Manga": FinalAssembler_Manga,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractivePanelCreator_Manga": "① Create Panel Layout (Interactive)",
    "MangaPanelDetector_Ultimate": "① Detect Panels by Color (Alternative)",
    "CropPanelForInpaint_Advanced": "② Crop Panel (Shape Aware)",
    "ConditionalLatentScaler_Final": "③ Conditionally Scale Latent",
    "AssembleAndProgress_Manga": "④ Assemble Panel (with Layout Index)",
    # --- Upscale Utils カテゴリ ---
    "LayoutExtractor_Ultimate": "U-② Extract Layout from Image",
    "LoadMangaFromOutput_Manga": "U-① Load Manga from OUTPUT", # <--- 新しいローダーの表示名
    "FinalAssembler_Manga": "U-③ Assemble Upscaled Panel", # <--- 表示名を変更
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Toolbox (with Upscale Utilities) ###")