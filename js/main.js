// /ComfyUI/custom_nodes/MangaInpaintToolbox/js/main.js

import { app } from "/scripts/app.js";

const PRESET_DIR_PATH = "extensions/MangaInpaintToolbox/presets";

console.log("### MangaInpaintToolbox JS: Script Loaded ###");

app.registerExtension({
    name: "MangaInpaintToolbox.UI",

    async nodeCreated(node) {
        
        // -------------------------------------------------------------------
        // --- UI Logic for: InteractivePanelCreator_Manga
        // -------------------------------------------------------------------
        if (node.comfyClass === "InteractivePanelCreator_Manga") {
            console.log(`★★★ SUCCESS: MangaInpaintToolbox JS is linked to node ${node.id} ★★★`);

            try {
                // --- 状態管理 ---
                let regions = [];
                let currentPolygonPoints = [];
                let isDrawing = false;
                let drawMode = "rect";
                let startPos = { x: 0, y: 0 };
                let currentRect = { x: 0, y: 0, w: 0, h: 0 };

                // --- ウィジェットの取得 ---
                const jsonWidget = node.widgets.find(w => w.name === "regions_json");
                const presetWidget = node.widgets.find(w => w.name === "preset");

                // --- UI要素の作成 ---
                const container = document.createElement("div");
                container.style.cssText = `display: flex; flex-direction: column; gap: 5px; padding: 5px;`;

                const modeSelector = document.createElement("div");
                modeSelector.style.cssText = `display: flex; gap: 10px; margin-bottom: 5px;`;
                ['rect', 'poly'].forEach(mode => {
                    const label = document.createElement("label");
                    label.style.cursor = "pointer";
                    const radio = document.createElement("input");
                    radio.type = "radio";
                    radio.name = `drawMode_${node.id}`;
                    radio.value = mode;
                    radio.checked = drawMode === mode;
                    radio.onchange = () => { drawMode = mode; finishPolygon(); };
                    label.appendChild(radio);
                    label.appendChild(document.createTextNode(mode.charAt(0).toUpperCase() + mode.slice(1)));
                    modeSelector.appendChild(label);
                });

                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");

                const buttonContainer = document.createElement("div");
                buttonContainer.style.cssText = `display: flex; gap: 5px; justify-content: space-between;`;
                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear";
                const undoButton = document.createElement("button");
                undoButton.textContent = "Undo";
                const finishPolyButton = document.createElement("button");
                finishPolyButton.textContent = "Finish Poly";

                buttonContainer.appendChild(clearButton);
                buttonContainer.appendChild(undoButton);
                buttonContainer.appendChild(finishPolyButton);
                
                // --- UIの組み立て ---
                container.appendChild(modeSelector);
                container.appendChild(canvas);
                container.appendChild(buttonContainer);

                const customWidget = node.addDOMWidget("interactive_canvas", "div", container, { serialize: false });

                // --- ヘルパー関数 ---
                const updateUIVisibility = () => {
                    const isManual = presetWidget.value === "(Manual Canvas)";
                    container.style.display = isManual ? "flex" : "none";
                    node.computeSize();
                };
                
                const syncDataToWidget = () => {
                    const jsonString = JSON.stringify(regions);
                    if (jsonWidget.value !== jsonString) {
                        jsonWidget.value = jsonString;
                        app.graph.setDirtyCanvas(true, true);
                    }
                };

                const finishPolygon = () => {
                    if (currentPolygonPoints.length >= 3) {
                        regions.push({ type: "poly", points: [...currentPolygonPoints] });
                        syncDataToWidget();
                    }
                    currentPolygonPoints = [];
                    redraw();
                };

                const redraw = () => {
                    const w = node.widgets.find(w => w.name === "width").value;
                    const h = node.widgets.find(w => w.name === "height").value;
                    canvas.width = w;
                    canvas.height = h;
                    canvas.style.backgroundColor = "#222";
                    canvas.style.border = "1px solid #555";
                    
                    ctx.clearRect(0, 0, w, h);
                    ctx.lineWidth = 2;
                    
                    regions.forEach((region, i) => {
                        ctx.strokeStyle = `hsla(${(i*40)%360}, 90%, 60%, 0.8)`;
                        if (region.type === 'rect') {
                            ctx.strokeRect(region.x, region.y, region.w, region.h);
                        } else if (region.type === 'poly') {
                            ctx.beginPath();
                            region.points.forEach((p, j) => j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                            ctx.closePath();
                            ctx.stroke();
                        }
                    });

                    if (isDrawing && drawMode === 'rect') {
                        ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
                        ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
                    }

                    if (currentPolygonPoints.length > 0) {
                        ctx.strokeStyle = "rgba(0, 255, 0, 0.8)";
                        ctx.beginPath();
                        currentPolygonPoints.forEach((p, j) => j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                        ctx.stroke();
                    }
                };

                const getScaledCoords = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    if (rect.width === 0) return { x: 0, y: 0 };
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    return {
                        x: Math.round(Math.max(0, Math.min(canvas.width, (e.clientX - rect.left) * scaleX))),
                        y: Math.round(Math.max(0, Math.min(canvas.height, (e.clientY - rect.top) * scaleY))),
                    };
                };
                
                const loadPresetAndRedraw = async (presetFilename) => {
                    if (presetFilename === "(Manual Canvas)") {
                        try {
                           regions = JSON.parse(jsonWidget.value || "[]");
                        } catch(e) { regions = []; }
                        redraw();
                        return;
                    }
                    
                    try {
                        const response = await fetch(`/${PRESET_DIR_PATH}/${presetFilename}`);
                        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                        const presetData = await response.json();
                        regions = presetData;
                        jsonWidget.value = JSON.stringify(regions);
                        redraw();
                    } catch (error) {
                        console.error(`Failed to load preset ${presetFilename}:`, error);
                        regions = [];
                        jsonWidget.value = "[]";
                        redraw();
                    }
                };

                // --- イベントリスナー ---
                const originalPresetCallback = presetWidget.callback;
                presetWidget.callback = function(value) {
                    updateUIVisibility();
                    if (value !== "(Manual Canvas)") {
                        loadPresetAndRedraw(value);
                    } else {
                        try {
                           regions = JSON.parse(jsonWidget.value || "[]");
                        } catch(e) { regions = []; }
                        redraw();
                    }
                    if(originalPresetCallback) {
                        return originalPresetCallback.apply(this, arguments);
                    }
                };

                const handleMouseMove = e => {
                    if (!isDrawing) return;
                    const coords = getScaledCoords(e);
                    currentRect.w = Math.abs(startPos.x - coords.x);
                    currentRect.h = Math.abs(startPos.y - coords.y);
                    currentRect.x = Math.min(startPos.x, coords.x);
                    currentRect.y = Math.min(startPos.y, coords.y);
                    redraw();
                };

                const handleMouseUp = () => {
                    if (!isDrawing) return;
                    isDrawing = false;
                    window.removeEventListener("mousemove", handleMouseMove);
                    window.removeEventListener("mouseup", handleMouseUp);
                    if (currentRect.w > 5 && currentRect.h > 5) {
                        regions.push({ type: 'rect', ...currentRect });
                        syncDataToWidget();
                    }
                    redraw();
                };

                canvas.addEventListener("mousedown", e => {
                    if (drawMode === 'rect') {
                        isDrawing = true;
                        startPos = getScaledCoords(e);
                        currentRect = { x: startPos.x, y: startPos.y, w: 0, h: 0 };
                        window.addEventListener("mousemove", handleMouseMove);
                        window.addEventListener("mouseup", handleMouseUp);
                    } else { // poly mode
                        currentPolygonPoints.push(getScaledCoords(e));
                        redraw();
                    }
                });

                clearButton.onclick = () => { finishPolygon(); regions = []; syncDataToWidget(); redraw(); };
                undoButton.onclick = () => { finishPolygon(); regions.pop(); syncDataToWidget(); redraw(); };
                finishPolyButton.onclick = finishPolygon;
                
                const originalOnPropertyChanged = node.onPropertyChanged;
                node.onPropertyChanged = function(name, value) {
                    if(originalOnPropertyChanged) {
                        originalOnPropertyChanged.apply(this, arguments);
                    }
                    if (name === 'width' || name === 'height') {
                         redraw();
                    }
                };
                
                // --- 初期化処理 ---
                if (jsonWidget.value) {
                    try { regions = JSON.parse(jsonWidget.value); } catch(e) { regions = []; }
                }

                updateUIVisibility();
                redraw();

            } catch (error) {
                console.error("### MangaInpaintToolbox JS Error in InteractivePanelCreator_Manga ###", error);
            }
        }
        
        // -------------------------------------------------------------------
        // --- UI Logic for: LoadMangaFromOutput_Manga (Previewer)
        // -------------------------------------------------------------------
        if (node.comfyClass === "LoadMangaFromOutput_Manga") {
            try {
                const imageWidget = node.widgets.find(w => w.name === "image");
                if (!imageWidget) return;

                const previewContainer = document.createElement("div");
                previewContainer.style.cssText = `width: 100%; text-align: center;`;
                const previewImage = document.createElement("img");
                previewImage.style.cssText = `max-width: 100%; max-height: 300px; margin-top: 10px; display: none; object-fit: contain;`;
                previewContainer.appendChild(previewImage);

                node.addDOMWidget("preview", "div", previewContainer, { serialize: false });

                const updatePreview = (fileName) => {
                    if (fileName) {
                        const parts = fileName.replace(/\\/g, "/").split("/");
                        const filenameOnly = parts.pop();
                        const subfolder = parts.join("/");
                        
                        // Add a cache-busting query parameter
                        const src = `/view?filename=${encodeURIComponent(filenameOnly)}&type=output&subfolder=${encodeURIComponent(subfolder)}&t=${+new Date()}`;
                        
                        previewImage.src = src;
                        previewImage.style.display = "block";
                    } else {
                        previewImage.style.display = "none";
                    }
                };
                
                const originalCallback = imageWidget.callback;
                imageWidget.callback = function(value) {
                    updatePreview(value);
                    if (originalCallback) {
                        return originalCallback.apply(this, arguments);
                    }
                };

                // 初期表示
                updatePreview(imageWidget.value);

            } catch(error) {
                console.error("### MangaInpaintToolbox JS Error in LoadMangaFromOutput_Manga ###", error);
            }
        }

    }
});