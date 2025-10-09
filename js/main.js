// /ComfyUI/custom_nodes/MangaInpaintToolbox/js/main.js
// このコードでファイル全体を置き換えてください。(v3.5 Final Syntax Fix)

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PRESET_DIR_PATH = "extensions/MangaInpaintToolbox/presets";

console.log("### MangaInpaintToolbox JS: Script Loaded (v3.5 Final Syntax Fix) ###");

app.registerExtension({
    name: "MangaInpaintToolbox.IntegratedUI",

    async nodeCreated(node) {
        
        // -------------------------------------------------------------------
        // --- UI Logic for: InteractivePanelCreator_Manga
        // -------------------------------------------------------------------
        if (node.comfyClass === "InteractivePanelCreator_Manga") {
            try {
                let regions = [];
                let currentPolygonPoints = [];
                let isDrawing = false;
                let drawMode = "rect";
                let startPos = { x: 0, y: 0 };
                let currentRect = { x: 0, y: 0, w: 0, h: 0 };
                const jsonWidget = node.widgets.find(w => w.name === "regions_json");
                const presetWidget = node.widgets.find(w => w.name === "preset");
                const container = document.createElement("div");
                container.style.cssText = `display: flex; flex-direction: column; gap: 5px; padding: 5px;`;
                const modeSelector = document.createElement("div");
                modeSelector.style.cssText = `display: flex; gap: 10px; margin-bottom: 5px;`;
                ['rect', 'poly'].forEach(mode => { const label = document.createElement("label"); label.style.cursor = "pointer"; const radio = document.createElement("input"); radio.type = "radio"; radio.name = `drawMode_${node.id}`; radio.value = mode; radio.checked = drawMode === mode; radio.onchange = () => { drawMode = mode; finishPolygon(); }; label.appendChild(radio); label.appendChild(document.createTextNode(mode.charAt(0).toUpperCase() + mode.slice(1))); modeSelector.appendChild(label); });
                
                const canvas = document.createElement("canvas");
                canvas.style.width = "100%";
                canvas.style.height = "auto";
                canvas.style.display = "block";
                
                const ctx = canvas.getContext("2d");
                const buttonContainer = document.createElement("div");
                buttonContainer.style.cssText = `display: flex; gap: 5px; justify-content: space-between;`;
                const clearButton = document.createElement("button"); clearButton.textContent = "Clear";
                const undoButton = document.createElement("button"); undoButton.textContent = "Undo";
                const finishPolyButton = document.createElement("button"); finishPolyButton.textContent = "Finish Poly";
                buttonContainer.appendChild(clearButton); buttonContainer.appendChild(undoButton); buttonContainer.appendChild(finishPolyButton);
                container.appendChild(modeSelector); container.appendChild(canvas); container.appendChild(buttonContainer);
                const customWidget = node.addDOMWidget("interactive_canvas", "div", container, { serialize: false });
                
                const updateUIVisibility = () => { const isManual = presetWidget.value === "(Manual Canvas)"; container.style.display = isManual ? "flex" : "none"; node.computeSize(); };
                const syncDataToWidget = () => { const jsonString = JSON.stringify(regions); if (jsonWidget.value !== jsonString) { jsonWidget.value = jsonString; app.graph.setDirtyCanvas(true, true); } };
                const finishPolygon = () => { if (currentPolygonPoints.length >= 3) { regions.push({ type: "poly", points: [...currentPolygonPoints] }); syncDataToWidget(); } currentPolygonPoints = []; redraw(); };
                
                const redraw = () => {
                    if (!customWidget.element.parentElement) return;
                    const w = node.widgets.find(w => w.name === "width").value; const h = node.widgets.find(w => w.name === "height").value;
                    canvas.width = w; canvas.height = h;
                    canvas.style.backgroundColor = "#222"; canvas.style.border = "1px solid #555"; ctx.clearRect(0, 0, w, h); ctx.lineWidth = 2; regions.forEach((region, i) => { ctx.strokeStyle = `hsla(${(i*40)%360}, 90%, 60%, 0.8)`; if (region.type === 'rect') { ctx.strokeRect(region.x, region.y, region.w, region.h); } else if (region.type === 'poly') { ctx.beginPath(); region.points.forEach((p, j) => j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)); ctx.closePath(); ctx.stroke(); } }); if (isDrawing && drawMode === 'rect') { ctx.strokeStyle = "rgba(255, 0, 0, 0.8)"; ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h); } if (currentPolygonPoints.length > 0) { ctx.strokeStyle = "rgba(0, 255, 0, 0.8)"; ctx.beginPath(); currentPolygonPoints.forEach((p, j) => j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)); ctx.stroke(); } };
                
                const getScaledCoords = (e) => { const rect = canvas.getBoundingClientRect(); if (rect.width === 0) return { x: 0, y: 0 }; const scaleX = canvas.width / rect.width; const scaleY = canvas.height / rect.height; return { x: Math.round(Math.max(0, Math.min(canvas.width, (e.clientX - rect.left) * scaleX))), y: Math.round(Math.max(0, Math.min(canvas.height, (e.clientY - rect.top) * scaleY))) }; };
                const loadPresetAndRedraw = async (presetFilename) => { if (presetFilename === "(Manual Canvas)") { try { regions = JSON.parse(jsonWidget.value || "[]"); } catch(e) { regions = []; } redraw(); return; } try { const response = await fetch(`/${PRESET_DIR_PATH}/${presetFilename}`); if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`); const presetData = await response.json(); regions = presetData; jsonWidget.value = JSON.stringify(regions); redraw(); } catch (error) { console.error(`Failed to load preset ${presetFilename}:`, error); regions = []; jsonWidget.value = "[]"; redraw(); } };
                const originalPresetCallback = presetWidget.callback;
                presetWidget.callback = function(value) { updateUIVisibility(); if (value !== "(Manual Canvas)") { loadPresetAndRedraw(value); } else { try { regions = JSON.parse(jsonWidget.value || "[]"); } catch(e) { regions = []; } redraw(); } if(originalPresetCallback) { return originalPresetCallback.apply(this, arguments); } };
                const handleMouseMove = e => { if (!isDrawing) return; const coords = getScaledCoords(e); currentRect.w = Math.abs(startPos.x - coords.x); currentRect.h = Math.abs(startPos.y - coords.y); currentRect.x = Math.min(startPos.x, coords.x); currentRect.y = Math.min(startPos.y, coords.y); redraw(); };
                const handleMouseUp = () => { if (!isDrawing) return; isDrawing = false; window.removeEventListener("mousemove", handleMouseMove); window.removeEventListener("mouseup", handleMouseUp); if (currentRect.w > 5 && currentRect.h > 5) { regions.push({ type: 'rect', ...currentRect }); syncDataToWidget(); } redraw(); };
                canvas.addEventListener("mousedown", e => { if (drawMode === 'rect') { isDrawing = true; startPos = getScaledCoords(e); currentRect = { x: startPos.x, y: startPos.y, w: 0, h: 0 }; window.addEventListener("mousemove", handleMouseMove); window.addEventListener("mouseup", handleMouseUp); } else { currentPolygonPoints.push(getScaledCoords(e)); redraw(); } });
                clearButton.onclick = () => { finishPolygon(); regions = []; syncDataToWidget(); redraw(); };
                undoButton.onclick = () => { finishPolygon(); regions.pop(); syncDataToWidget(); redraw(); };
                finishPolyButton.onclick = finishPolygon;
                
                const originalOnPropertyChanged = node.onPropertyChanged;
                node.onPropertyChanged = function(name, value) { if(originalOnPropertyChanged) { originalOnPropertyChanged.apply(this, arguments); } if (name === 'width' || name === 'height') { redraw(); } };
                
                const originalOnResize = node.onResize;
                node.onResize = function() { originalOnResize?.apply(this, arguments); setTimeout(() => { redraw(); }, 0); };
                
                if (jsonWidget.value) { try { regions = JSON.parse(jsonWidget.value); } catch(e) { regions = []; } }
                updateUIVisibility();
                setTimeout(() => { redraw(); }, 0);
            } catch (error) { console.error("### MangaInpaintToolbox JS Error in InteractivePanelCreator_Manga ###", error); }
        }
        
        // -------------------------------------------------------------------
        // --- UI Logic for: LoadMangaFromOutput_Manga
        // -------------------------------------------------------------------
        if (node.comfyClass === "LoadMangaFromOutput_Manga") {
            try {
                const imageWidget = node.widgets.find(w => w.name === "image");
                if (!imageWidget) return;
                const mainContainer = document.createElement("div");
                mainContainer.style.cssText = `display: flex; flex-direction: column; align-items: center; width: 100%; gap: 8px;`;
                const previewImage = document.createElement("img");
                previewImage.style.cssText = `max-width: 100%; max-height: 300px; margin-top: 5px; display: none; object-fit: contain;`;
                const refreshButton = document.createElement("button");
                refreshButton.textContent = "Refresh List ↺";
                refreshButton.style.cssText = `width: 100%; padding: 5px; cursor: pointer;`;
                mainContainer.appendChild(previewImage);
                mainContainer.appendChild(refreshButton);
                node.addDOMWidget("manga_loader_ui", "div", mainContainer, { serialize: false });
                const updatePreview = (fileName) => {
                    if (fileName) {
                        const parts = fileName.replace(/\\/g, "/").split("/");
                        const filenameOnly = parts.pop();
                        const subfolder = parts.join("/");
                        const src = `/view?filename=${encodeURIComponent(filenameOnly)}&type=output&subfolder=${encodeURIComponent(subfolder)}&t=${+new Date()}`;
                        previewImage.src = src;
                        previewImage.style.display = "block";
                    } else { previewImage.style.display = "none"; }
                };
                refreshButton.addEventListener("click", async () => {
                    const originalText = refreshButton.textContent;
                    refreshButton.textContent = "Refreshing...";
                    refreshButton.disabled = true;
                    try {
                        const response = await api.fetchApi("/manga-toolbox/get-output-files");
                        if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                        const newFiles = await response.json();
                        const currentValue = imageWidget.value;
                        imageWidget.options.values = newFiles;
                        if (newFiles.includes(currentValue)) { imageWidget.value = currentValue; } 
                        else if (newFiles.length > 0) { imageWidget.value = newFiles[0]; }
                        if (imageWidget.callback) { imageWidget.callback(imageWidget.value); }
                    } catch (error) { console.error("Failed to refresh file list:", error); } 
                    finally { refreshButton.textContent = originalText; refreshButton.disabled = false; }
                });
                const originalCallback = imageWidget.callback;
                imageWidget.callback = function(value) { updatePreview(value); if (originalCallback) { return originalCallback.apply(this, arguments); } };
                updatePreview(imageWidget.value);
            } catch(error) { console.error("### MangaInpaintToolbox JS Error in LoadMangaFromOutput_Manga ###", error); }
        }

        // -------------------------------------------------------------------
        // --- UI Logic for: PanelArrangerForI2I_Manga (v3.5 Final Syntax Fix)
        // -------------------------------------------------------------------
        if (node.comfyClass === "PanelArrangerForI2I_Manga") {
            try {
                let regions = [], placedImages = [], currentPolygonPoints = [];
                let isDrawing = false, drawMode = "rect", isDragging = false, isResizing = false, isCropping = false, isDraggingCropBox = false;
                let startPos = { x: 0, y: 0 }, currentRect = { x: 0, y: 0, w: 0, h: 0 }, dragStartOffset = { x: 0, y: 0 };
                let selectedImageIndex = -1, cropImageIndex = -1, resizeHandle = null;

                const jsonWidget = node.widgets.find(w => w.name === "arrangement_json");
                
                const container = document.createElement("div");
                container.style.cssText = `display: flex; flex-direction: column; gap: 5px; padding: 5px;`;
                
                const modeSelector = document.createElement("div");
                modeSelector.style.cssText = `display: flex; gap: 10px; margin-bottom: 5px;`;
                ['rect', 'poly'].forEach(mode => {
                    const label = document.createElement("label"); label.style.cursor = "pointer";
                    const radio = document.createElement("input"); radio.type = "radio"; radio.name = `drawMode_${node.id}`; radio.value = mode; radio.checked = drawMode === mode;
                    radio.onchange = () => { drawMode = mode; finishPolygon(); };
                    label.appendChild(radio); label.appendChild(document.createTextNode(mode.charAt(0).toUpperCase() + mode.slice(1)));
                    modeSelector.appendChild(label);
                });

                const canvasContainer = document.createElement("div");
                canvasContainer.style.cssText = `position: relative; width: 100%; height: auto;`;
                const canvas = document.createElement("canvas");
                canvas.style.width = "100%";
                canvas.style.height = "auto";
                canvas.style.display = "block";
                const ctx = canvas.getContext("2d");
                
                const cropCanvas = document.createElement("canvas");
                cropCanvas.style.cssText = `position: absolute; top: 0; left: 0; pointer-events: none; display: none; z-index: 10; width: 100%; height: 100%;`;

                canvasContainer.append(canvas, cropCanvas);
                
                const buttonContainer = document.createElement("div");
                buttonContainer.style.cssText = `display: flex; gap: 5px; justify-content: flex-start; flex-wrap: wrap;`;
                
                const clearButton = document.createElement("button"); clearButton.textContent = "Clear Panels";
                const undoButton = document.createElement("button"); undoButton.textContent = "Undo Panel";
                const finishPolyButton = document.createElement("button"); finishPolyButton.textContent = "Finish Poly";
                const uploadButton = document.createElement("button"); uploadButton.textContent = "Add Image";
                const deleteImageButton = document.createElement("button"); deleteImageButton.textContent = "Delete Image";
                const imageUploader = document.createElement("input"); imageUploader.type = "file"; imageUploader.accept = "image/png, image/jpeg, image/webp"; imageUploader.style.display = "none";
                const cropButton = document.createElement("button"); cropButton.textContent = "Crop/Resize Selected";
                const cropConfirmButton = document.createElement("button"); cropConfirmButton.textContent = "Confirm Crop"; cropConfirmButton.style.display = "none";
                const cropCancelButton = document.createElement("button"); cropCancelButton.textContent = "Cancel Crop"; cropCancelButton.style.display = "none";

                buttonContainer.append(clearButton, undoButton, finishPolyButton, uploadButton, deleteImageButton, cropButton, cropConfirmButton, cropCancelButton);
                container.append(imageUploader, modeSelector, canvasContainer, buttonContainer);
                const customWidget = node.addDOMWidget("interactive_canvas_i2i", "div", container, { serialize: false });
                
                const getResizeHandles = (box) => {
                    const handleSize = 8; const hs = handleSize / 2;
                    return {
                        'top-left':     { x: box.x - hs, y: box.y - hs, w: handleSize, h: handleSize, cursor: 'nwse-resize' },
                        'top-right':    { x: box.x + box.w - hs, y: box.y - hs, w: handleSize, h: handleSize, cursor: 'nesw-resize' },
                        'bottom-left':  { x: box.x - hs, y: box.y + box.h - hs, w: handleSize, h: handleSize, cursor: 'nesw-resize' },
                        'bottom-right': { x: box.x + box.w - hs, y: box.y + box.h - hs, w: handleSize, h: handleSize, cursor: 'nwse-resize' },
                        'top-middle':   { x: box.x + box.w / 2 - hs, y: box.y - hs, w: handleSize, h: handleSize, cursor: 'ns-resize' },
                        'bottom-middle':{ x: box.x + box.w / 2 - hs, y: box.y + box.h - hs, w: handleSize, h: handleSize, cursor: 'ns-resize' },
                        'left-middle':  { x: box.x - hs, y: box.y + box.h / 2 - hs, w: handleSize, h: handleSize, cursor: 'ew-resize' },
                        'right-middle': { x: box.x + box.w - hs, y: box.y + box.h / 2 - hs, w: handleSize, h: handleSize, cursor: 'ew-resize' },
                    };
                };
                const getHandleAtPos = (box, x, y) => {
                    const handles = getResizeHandles(box);
                    for (const [name, rect] of Object.entries(handles)) { if (x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h) return name; }
                    return null;
                };
                const syncDataToWidget = () => {
                    const serializableImages = placedImages.map(pImg => ({ src: pImg.fileInfo.filename, x: pImg.x, y: pImg.y, w: pImg.w, h: pImg.h, cropRect: pImg.cropRect }));
                    const jsonString = JSON.stringify({ regions, images: serializableImages });
                    if (jsonWidget.value !== jsonString) { jsonWidget.value = jsonString; app.graph.setDirtyCanvas(true, true); }
                };
                const redraw = () => {
                    if (!customWidget.element.parentElement) return;
                    const w = node.widgets.find(w => w.name === "width").value;
                    const h = node.widgets.find(w => w.name === "height").value;
                    canvas.width = w; canvas.height = h;
                    ctx.fillStyle = "#222"; ctx.fillRect(0, 0, w, h); ctx.lineWidth = 2;
                    regions.forEach((region, i) => {
                        ctx.strokeStyle = `hsla(${(i * 40) % 360}, 90%, 60%, 0.8)`;
                        if (region.type === 'rect') ctx.strokeRect(region.x, region.y, region.w, region.h);
                        else if (region.type === 'poly') { ctx.beginPath(); region.points.forEach((p, j) => j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)); ctx.closePath(); ctx.stroke(); }
                    });
                    placedImages.forEach((pImg, i) => {
                        try {
                            ctx.drawImage(pImg.originalImg, pImg.cropRect.x, pImg.cropRect.y, pImg.cropRect.w, pImg.cropRect.h, pImg.x, pImg.y, pImg.w, pImg.h);
                        } catch(e) {
                            try { ctx.drawImage(pImg.originalImg, pImg.x, pImg.y, pImg.w, pImg.h); } catch(err) {}
                        }
                        if (i === selectedImageIndex && !isCropping) {
                            ctx.strokeStyle = "rgba(255, 255, 0, 0.9)"; ctx.strokeRect(pImg.x, pImg.y, pImg.w, pImg.h);
                            ctx.fillStyle = "rgba(255, 255, 0, 0.9)";
                            Object.values(getResizeHandles(pImg)).forEach(rect => ctx.fillRect(rect.x, rect.y, rect.w, rect.h));
                        }
                    });
                    if (isDrawing && drawMode === 'rect') { ctx.strokeStyle = "rgba(255, 0, 0, 0.8)"; ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h); }
                    if (currentPolygonPoints.length > 0) { ctx.strokeStyle = "rgba(0, 255, 0, 0.8)"; ctx.beginPath(); currentPolygonPoints.forEach((p, j) => j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)); ctx.stroke(); }
                };
                const drawCropUI = () => {
                    if (!isCropping || cropImageIndex === -1) { cropCanvas.style.display = "none"; return; }
                    cropCanvas.style.display = "block";
                    const w = canvas.width; const h = canvas.height;
                    cropCanvas.width = w; cropCanvas.height = h;
                    const cropCtx = cropCanvas.getContext("2d");
                    cropCtx.clearRect(0, 0, w, h);
                    cropCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
                    cropCtx.fillRect(0, 0, w, h);
                    const pImg = placedImages[cropImageIndex];
                    cropCtx.drawImage(pImg.originalImg, pImg.cropRect.x, pImg.cropRect.y, pImg.cropRect.w, pImg.cropRect.h, pImg.x, pImg.y, pImg.w, pImg.h);
                    cropCtx.strokeStyle = "rgba(255, 255, 0, 1)"; cropCtx.lineWidth = 2;
                    cropCtx.strokeRect(pImg.tempCropBox.x, pImg.tempCropBox.y, pImg.tempCropBox.w, pImg.tempCropBox.h);
                    cropCtx.fillStyle = "rgba(255, 255, 0, 0.9)";
                    Object.values(getResizeHandles(pImg.tempCropBox)).forEach(rect => cropCtx.fillRect(rect.x, rect.y, rect.w, rect.h));
                };
                const finishPolygon = () => { if (currentPolygonPoints.length >= 3) { regions.push({ type: "poly", points: [...currentPolygonPoints] }); } currentPolygonPoints = []; syncDataToWidget(); redraw(); };
                const getScaledCoords = (e) => {
                    const rect = e.target.getBoundingClientRect();
                    if (rect.width === 0) return { x: 0, y: 0 };
                    const scaleX = e.target.width / rect.width;
                    const scaleY = e.target.height / rect.height;
                    return { x: Math.round(Math.max(0, (e.clientX - rect.left) * scaleX)), y: Math.round(Math.max(0, (e.clientY - rect.top) * scaleY)) };
                };
                const handleMouseDown = e => {
                    const coords = getScaledCoords(e);
                    if (isCropping) {
                        const pImg = placedImages[cropImageIndex];
                        resizeHandle = getHandleAtPos(pImg.tempCropBox, coords.x, coords.y);
                        if(resizeHandle) { isResizing = true; } 
                        else if (coords.x > pImg.tempCropBox.x && coords.x < pImg.tempCropBox.x + pImg.tempCropBox.w && coords.y > pImg.tempCropBox.y && coords.y < pImg.tempCropBox.y + pImg.tempCropBox.h) {
                            isDraggingCropBox = true;
                            dragStartOffset = { x: coords.x - pImg.tempCropBox.x, y: coords.y - pImg.tempCropBox.y };
                        }
                        return;
                    }
                    if (selectedImageIndex !== -1) {
                        const handle = getHandleAtPos(placedImages[selectedImageIndex], coords.x, coords.y);
                        if (handle) { isResizing = true; resizeHandle = handle; return; }
                    }
                    let clickedOnImage = false;
                    for (let i = placedImages.length - 1; i >= 0; i--) {
                        const pImg = placedImages[i];
                        if (coords.x >= pImg.x && coords.x <= pImg.x + pImg.w && coords.y >= pImg.y && coords.y <= pImg.y + pImg.h) {
                            if(selectedImageIndex !== i) { selectedImageIndex = i; redraw(); }
                            isDragging = true;
                            dragStartOffset = { x: coords.x - pImg.x, y: coords.y - pImg.y };
                            clickedOnImage = true;
                            break;
                        }
                    }
                    if(!clickedOnImage && selectedImageIndex !== -1) { selectedImageIndex = -1; redraw(); }
                    if (!isDragging) {
                        if (drawMode === 'rect') { isDrawing = true; startPos = coords; currentRect = { x: startPos.x, y: startPos.y, w: 0, h: 0 }; } 
                        else { currentPolygonPoints.push(coords); redraw(); }
                    }
                };
                const handleMouseMove = e => {
                    const coords = getScaledCoords(e);
                    let cursor = 'default';
                    if (isCropping) {
                        const pImg = placedImages[cropImageIndex];
                        if(isResizing) {
                            const box = pImg.tempCropBox; const lastX = box.x + box.w; const lastY = box.y + box.h;
                            if (resizeHandle.includes('right')) box.w = Math.max(10, coords.x - box.x);
                            if (resizeHandle.includes('bottom')) box.h = Math.max(10, coords.y - box.y);
                            if (resizeHandle.includes('left')) { box.w = Math.max(10, lastX - coords.x); box.x = coords.x; }
                            if (resizeHandle.includes('top')) { box.h = Math.max(10, lastY - coords.y); box.y = coords.y; }
                        } else if(isDraggingCropBox) {
                            pImg.tempCropBox.x = coords.x - dragStartOffset.x; pImg.tempCropBox.y = coords.y - dragStartOffset.y;
                        } else {
                            const handle = getHandleAtPos(pImg.tempCropBox, coords.x, coords.y);
                            if(handle) cursor = getResizeHandles(pImg.tempCropBox)[handle].cursor;
                            else if(coords.x > pImg.tempCropBox.x && coords.x < pImg.tempCropBox.x + pImg.tempCropBox.w && coords.y > pImg.tempCropBox.y && coords.y < pImg.tempCropBox.y + pImg.tempCropBox.h) { cursor = 'move'; }
                        }
                        e.target.style.cursor = cursor; drawCropUI();
                    } else if (isResizing) {
                        const pImg = placedImages[selectedImageIndex];
                        const lastX = pImg.x + pImg.w; const lastY = pImg.y + pImg.h; const aspect = pImg.cropRect.w / pImg.cropRect.h;
                        if (resizeHandle.includes('right')) pImg.w = Math.max(10, coords.x - pImg.x);
                        if (resizeHandle.includes('bottom')) pImg.h = Math.max(10, coords.y - pImg.y);
                        if (resizeHandle.includes('left')) { pImg.w = Math.max(10, lastX - coords.x); pImg.x = coords.x; }
                        if (resizeHandle.includes('top')) { pImg.h = Math.max(10, lastY - coords.y); pImg.y = coords.y; }
                        if (e.shiftKey) {
                            if (resizeHandle.includes('left') || resizeHandle.includes('right')) pImg.h = pImg.w / aspect; else pImg.w = pImg.h * aspect;
                        }
                        redraw();
                    } else if (isDragging) {
                        const pImg = placedImages[selectedImageIndex];
                        pImg.x = coords.x - dragStartOffset.x; pImg.y = coords.y - dragStartOffset.y; redraw();
                    } else if (isDrawing) {
                        currentRect.w = Math.abs(startPos.x - coords.x); currentRect.h = Math.abs(startPos.y - coords.y);
                        currentRect.x = Math.min(startPos.x, coords.x); currentRect.y = Math.min(startPos.y, coords.y); redraw();
                    } else {
                        if (selectedImageIndex !== -1) {
                           const handle = getHandleAtPos(placedImages[selectedImageIndex], coords.x, coords.y);
                           if(handle) cursor = getResizeHandles(placedImages[selectedImageIndex])[handle].cursor;
                        }
                        e.target.style.cursor = cursor;
                    }
                };
                const handleMouseUp = () => {
                    if (isResizing || isDragging || isDrawing || isDraggingCropBox) {
                        isResizing = false; isDragging = false; isDrawing = false; isDraggingCropBox = false;
                        if (!isCropping) {
                            if (currentRect.w > 5 && currentRect.h > 5) { regions.push({ type: 'rect', ...currentRect }); }
                            currentRect = {w:0, h:0}; syncDataToWidget();
                        }
                        redraw();
                    }
                };
                
                canvas.addEventListener("mousedown", handleMouseDown);
                canvas.addEventListener("mousemove", handleMouseMove);
                window.addEventListener("mouseup", handleMouseUp);
                cropCanvas.addEventListener("mousedown", handleMouseDown);
                cropCanvas.addEventListener("mousemove", handleMouseMove);

                clearButton.onclick = () => { finishPolygon(); regions = []; syncDataToWidget(); redraw(); };
                undoButton.onclick = () => { finishPolygon(); regions.pop(); syncDataToWidget(); redraw(); };
                finishPolyButton.onclick = finishPolygon;
                uploadButton.onclick = () => imageUploader.click();
                deleteImageButton.onclick = () => {
                    if (selectedImageIndex !== -1) { placedImages.splice(selectedImageIndex, 1); selectedImageIndex = -1; syncDataToWidget(); redraw(); }
                };
                imageUploader.onchange = async (e) => {
                    if (!e.target.files.length) return;
                    const file = e.target.files[0]; const formData = new FormData(); formData.append("image", file);
                    uploadButton.textContent = "Uploading...";
                    try {
                        const response = await api.fetchApi("/manga-toolbox/upload-image", { method: "POST", body: formData });
                        if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
                        const fileInfo = await response.json();
                        const reader = new FileReader();
                        reader.onload = (event) => {
                            const img = new Image();
                            img.onload = () => {
                                const initialScale = Math.min(256 / img.width, 256 / img.height, 1);
                                placedImages.push({ fileInfo, img, originalImg: img, x: 10, y: 10, w: img.width * initialScale, h: img.height * initialScale, cropRect: { x: 0, y: 0, w: img.width, h: img.height } });
                                syncDataToWidget(); redraw();
                            };
                            img.src = event.target.result;
                        };
                        reader.readAsDataURL(file);
                    } catch(error) { console.error("### MangaInpaintToolbox JS: Image upload error ###", error); alert("Image upload failed. See browser console (F12) for details."); } 
                    finally { uploadButton.textContent = "Add Image"; imageUploader.value = ""; }
                };
                cropButton.onclick = () => {
                    if (selectedImageIndex === -1) { alert("Please select an image to crop."); return; }
                    isCropping = true; cropImageIndex = selectedImageIndex;
                    const pImg = placedImages[cropImageIndex];
                    pImg.tempCropBox = { x: pImg.x, y: pImg.y, w: pImg.w, h: pImg.h, };
                    cropButton.style.display = "none"; deleteImageButton.style.display = "none";
                    cropConfirmButton.style.display = "inline-block"; cropCancelButton.style.display = "inline-block";
                    canvas.style.pointerEvents = "none"; cropCanvas.style.pointerEvents = "auto";
                    drawCropUI();
                };
                const exitCropMode = () => {
                    isCropping = false; cropImageIndex = -1;
                    cropButton.style.display = "inline-block"; deleteImageButton.style.display = "inline-block";
                    cropConfirmButton.style.display = "none"; cropCancelButton.style.display = "none";
                    cropCanvas.style.display = "none"; canvas.style.pointerEvents = "auto"; canvas.style.cursor = 'default';
                    redraw();
                };
                cropCancelButton.onclick = exitCropMode;
                
                cropConfirmButton.onclick = async () => {
                    if (cropImageIndex === -1) return;
                    const pImg = placedImages[cropImageIndex];
                    try {
                        const sourceToDisplayRatioX = pImg.cropRect.w / pImg.w;
                        const sourceToDisplayRatioY = pImg.cropRect.h / pImg.h;
                        const newCropX = pImg.cropRect.x + (pImg.tempCropBox.x - pImg.x) * sourceToDisplayRatioX;
                        const newCropY = pImg.cropRect.y + (pImg.tempCropBox.y - pImg.y) * sourceToDisplayRatioY;
                        const newCropW = pImg.tempCropBox.w * sourceToDisplayRatioX;
                        const newCropH = pImg.tempCropBox.h * sourceToDisplayRatioY;
                        const sx = Math.max(0, Math.round(newCropX));
                        const sy = Math.max(0, Math.round(newCropY));
                        const sw = Math.max(1, Math.round(newCropW));
                        const sh = Math.max(1, Math.round(newCropH));
                        const off = document.createElement("canvas");
                        off.width = sw; off.height = sh;
                        const offCtx = off.getContext("2d");
                        offCtx.drawImage(pImg.originalImg, sx, sy, sw, sh, 0, 0, sw, sh);
                        const blob = await new Promise(resolve => off.toBlob(resolve, "image/png"));
                        const formData = new FormData();
                        const suggestedName = `cropped_${pImg.fileInfo.filename.replace(/[^a-zA-Z0-9.\-_]/g, "_")}`;
                        formData.append("image", blob, suggestedName);
                        cropConfirmButton.textContent = "Processing...";
                        cropConfirmButton.disabled = true;
                        const uploadResp = await api.fetchApi("/manga-toolbox/upload-image", { method: "POST", body: formData });
                        if (!uploadResp.ok) throw new Error(`Upload failed: ${uploadResp.statusText}`);
                        const newFileInfo = await uploadResp.json();
                        const newImg = new Image();
                        newImg.onload = () => {
                            pImg.originalImg = newImg;
                            pImg.cropRect = { x: 0, y: 0, w: sw, h: sh };
                            pImg.x = pImg.tempCropBox.x;
                            pImg.y = pImg.tempCropBox.y;
                            pImg.w = pImg.tempCropBox.w;
                            pImg.h = pImg.tempCropBox.h;
                            pImg.fileInfo = newFileInfo;
                            delete pImg.tempCropBox;
                            syncDataToWidget();
                            exitCropMode();
                        };
                        newImg.src = off.toDataURL("image/png");
                    } catch (err) {
                        console.error("Crop/Upload error:", err);
                        alert("Crop failed. See console for details.");
                    } finally {
                        cropConfirmButton.textContent = "Confirm Crop";
                        cropConfirmButton.disabled = false;
                    }
                };

                const restoreStateFromWidget = () => {
                    try { const data = JSON.parse(jsonWidget.value || "{}"); regions = data.regions || []; placedImages = []; } catch (e) { regions = []; placedImages = []; }
                };
                
                restoreStateFromWidget();
                
                const originalComputeSize = node.computeSize;
                node.computeSize = function(out) {
                    const size = originalComputeSize.apply(this, arguments);
                    if (customWidget && customWidget.element && customWidget.element.offsetHeight) {
                        size[1] += customWidget.element.offsetHeight;
                    }
                    return size;
                };
                const originalOnResize = node.onResize;
                node.onResize = function() {
                    originalOnResize?.apply(this, arguments);
                    setTimeout(() => { redraw(); drawCropUI(); }, 0);
                };
                const originalOnPropertyChanged = node.onPropertyChanged;
                node.onPropertyChanged = function(name, value) {
                    originalOnPropertyChanged?.apply(this, arguments);
                    if (name === 'width' || name === 'height') { redraw(); drawCropUI(); }
                };
                setTimeout(() => { redraw(); node.setDirtyCanvas(true, true); }, 200);
                
            } catch (error) { console.error("### MangaInpaintToolbox JS Error in PanelArrangerForI2I_Manga ###", error); }
        }
    }
});