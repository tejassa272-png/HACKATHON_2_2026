import cv2
import numpy as np
import time
import tempfile
import os
import gc
import psutil # NEW: Hardware Telemetry
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import imageio

# ==========================================
# ⚙️ EDGE NODE INITIALIZATION
# ==========================================
MODEL_PATH = "bdd_drivable_opt.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']

print(f"\n[E.V.A. SYSTEM] Booting Perception Node...")
print(f"[E.V.A. SYSTEM] Hardware Target: {providers[0]}")

ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
IMG_H, IMG_W = 512, 896

app = FastAPI(title="E.V.A. Perception Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # NEW: Expose RAM Telemetry to the browser
    expose_headers=["X-Inference-Time", "X-Video-Metrics", "X-RAM-Usage"] 
)

TEMP_DIR = tempfile.gettempdir()

@app.get("/health")
async def health_check():
    return {"status": "ONLINE", "model": "DeepLabV3+_MobileNetV3", "device": providers[0]}

# ==========================================
# 📸 IMAGE / LIVE FRAME ENDPOINT
# ==========================================
@app.post("/segment_image")
async def process_image_api(file: UploadFile = File(...), threshold: float = Form(0.5)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        orig_h, orig_w = img_bgr.shape[:2]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_W, IMG_H))
        img_normalized = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std

        ort_input = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)
        t0 = time.time()
        preds = ort_session.run(None, {ort_session.get_inputs()[0].name: ort_input})[0]
        inference_time = (time.time() - t0) * 1000

        # NEW: Dynamic Threshold Math
        prob = 1.0 / (1.0 + np.exp(-preds[0, 0]))
        mask = (prob > threshold).astype(np.uint8)
        
        img_resized_bgr = cv2.resize(img_bgr, (IMG_W, IMG_H))
        overlay = img_resized_bgr.copy()
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array([100, 255, 0]) * 0.5 
        final_output = cv2.resize(overlay, (orig_w, orig_h))

        out_path = os.path.join(TEMP_DIR, f"out_{int(time.time()*1000)}.jpg")
        cv2.imwrite(out_path, final_output)
        
        # NEW: Read exact RAM footprint of this Python process
        process_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        return FileResponse(out_path, headers={"X-Inference-Time": str(inference_time), "X-RAM-Usage": f"{process_ram:.1f}"})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==========================================
# 🎥 TEMPORAL VIDEO ENDPOINT
# ==========================================
@app.post("/segment_video")
async def process_video_api(file: UploadFile = File(...), threshold: float = Form(0.5)):
    try:
        in_path = os.path.join(TEMP_DIR, f"in_{int(time.time())}.mp4")
        with open(in_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out_path = os.path.join(TEMP_DIR, f"out_{int(time.time())}.mp4")
        
        writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None)

        MAX_FRAMES = int(fps * 5) 
        frames_processed = 0
        start_time = time.time()

        while cap.isOpened() and frames_processed < MAX_FRAMES:
            ret, frame_bgr = cap.read()
            if not ret: break
            
            orig_h, orig_w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(frame_rgb, (IMG_W, IMG_H))
            
            img_normalized = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_normalized - mean) / std
            
            ort_input = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)
            preds = ort_session.run(None, {ort_session.get_inputs()[0].name: ort_input})[0]

            # NEW: Dynamic Threshold Math
            mask = (1.0 / (1.0 + np.exp(-preds[0, 0])) > threshold).astype(np.uint8)
            overlay_rgb = img_resized.copy()
            overlay_rgb[mask == 1] = overlay_rgb[mask == 1] * 0.5 + np.array([0, 255, 100]) * 0.5
            
            writer.append_data(cv2.resize(overlay_rgb, (orig_w, orig_h)).astype(np.uint8))
            frames_processed += 1

        cap.release()
        writer.close() 
        gc.collect() 

        # NEW: Inject RAM Telemetry into the video metrics string
        process_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        total_time = time.time() - start_time
        actual_fps = frames_processed / total_time if total_time > 0 else 0
        
        metrics_str = f"Time: {total_time:.1f}s | Speed: {actual_fps:.1f} FPS | RAM: {process_ram:.1f} MB"
        
        return FileResponse(out_path, media_type="video/mp4", headers={"X-Video-Metrics": metrics_str})

    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)