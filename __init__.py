# /ComfyUI/custom_nodes/MangaInpaintToolbox/__init__.py

from .nodes import (
    InteractivePanelCreator,
    MangaPanelDetector_Ultimate,
    CropPanelForInpaint_Advanced,
    ConditionalLatentScaler_Final,
    AssembleAndProgress,
)

# このノードがWeb UI用のファイルを持つことをComfyUIに伝える
WEB_DIRECTORY = "./js"

# HTMLに直接JSファイルをインポートさせる
WEB_EXTENSIONS = {
    "MangaInpaintToolbox": "/extensions/MangaInpaintToolbox/main.js",
}

NODE_CLASS_MAPPINGS = {
    "InteractivePanelCreator_Manga": InteractivePanelCreator,
    "MangaPanelDetector_Ultimate": MangaPanelDetector_Ultimate,
    "CropPanelForInpaint_Advanced": CropPanelForInpaint_Advanced,
    "ConditionalLatentScaler_Final": ConditionalLatentScaler_Final,
    "AssembleAndProgress_Manga": AssembleAndProgress,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractivePanelCreator_Manga": "① Create Panel Layout (Interactive)",
    "MangaPanelDetector_Ultimate": "① Detect Panels by Color (Alternative)",
    "CropPanelForInpaint_Advanced": "② Crop Panel (Shape Aware)",
    "ConditionalLatentScaler_Final": "③ Conditionally Scale Latent",
    "AssembleAndProgress_Manga": "④ Assemble Panel (with Progress)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("### Loading: Manga Inpaint Toolbox (with Interactive Creator & Progress Assembler) ###")