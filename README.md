# E.V.A. (Edge Vision Architecture)
**Level 4 Autonomous Vehicle Perception Node | Offline Edge Compute Prototype**

![E.V.A. Dashboard](https://img.shields.io/badge/Status-Live_Telemetry-10b981?style=for-the-badge)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-C++_Accelerated-3b82f6?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Async_Backend-009688?style=for-the-badge)
![WebRTC](https://img.shields.io/badge/Sensor-WebRTC_Live-ef4444?style=for-the-badge)

E.V.A. is a production-grade, entirely offline perception node designed for Level 4 autonomous path planning. It tackles a core safety hazard in autonomous driving: **Cloud Dependency.** By aggressively optimizing for Edge Compute, E.V.A. drops the massive latency and internet-dependency of traditional ResNet/SegFormer models, achieving stable real-time tensor processing strictly on local CPU hardware with a sub-300MB memory footprint.

## Core Engineering Architecture

Our solution specifically addresses the **Binary Classification of Drivable vs. Non-Drivable Space** using a decoupled, full-stack ML-Ops pipeline.

### 1. The Perception Engine (`bdd_drivable_opt.onnx`)
* **Task:** Binary Semantic Segmentation (Drivable Free-Space Estimation).
* **Architecture:** DeepLabV3+ with a lightweight **MobileNetV3** backbone, trained on the BDD100K dataset.
* **Edge Optimization:** Exported to a static ONNX computation graph. 32-bit float matrix multiplication is strictly enforced at the tensor level to prevent memory-type crashing on standard local hardware.

### 2. The API Gateway (`api.py`)
* **Framework:** Asynchronous FastAPI.
* **Temporal Bypass:** Integrated `imageio` H.264 video processing to completely bypass standard Windows OpenCV codec failures and buffer overloads.
* **Hardware Telemetry:** Integrates `psutil` to track exact physical RAM consumption and injects it into the HTTP headers for real-time monitoring.
* **Memory Management:** Aggressive Python garbage collection (`gc.collect()`) deployed post-inference to prevent memory leaks during continuous spatial evaluation.

### 3. The Telemetry Dashboard (`web.html`)
* **Zero-Dependency UI:** 100% Vanilla JavaScript and CSS.
* **Dynamic Probability Thresholding:** A live UI slider allows engineers to adjust the neural network's strictness (`0.1` to `0.9`) in real-time, actively mitigating Out-of-Distribution (OOD) indoor hallucinations.
* **Live Sensor Feed:** WebRTC integration hijacks local optical hardware (webcams) for continuous, real-time spatial evaluation directly through the browser.

* **Validation Metric:** Achieves an outstanding **92.56% mIoU** on the BDD100K validation split. The architecture maintains strict mathematical boundary precision for drivable space while guaranteeing stable, real-time inference on low-power local hardware.


## 4. PHOTO ANALYSIS DEMO
![Photo_Demo-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/7c886444-29dc-4ddc-a339-f4ed4541ce26)


## 5. VIDEO ANALYSIS DEMO
![Video_Demo-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/127c0020-eb92-4463-90e6-d3574ce9d9f9)


# How to Run The Code

**1. Clone the repository:**
```bash
git clone https://github.com/tejassa272-png/HACKATHON_2_2026.git
cd HACKATHON_2_2026
```
**2. Install the dependencies:**
```bash
pip install fastapi uvicorn onnxruntime opencv-python numpy imageio imageio-ffmpeg psutil python-multipart
```
**3. Run The api.py File And Do not Close It:**
```bash
python api.py
```
**4. Open the Interface:**
* Navigate to the project folder using your computer's file explorer and double-click the web.html file to open the live dashboard in your web browser.
* Or in the vscode click on open with live server.
