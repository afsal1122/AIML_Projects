# 🖐️ Hand Gesture AI Mouse Control System  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-yellow)
![Autopy](https://img.shields.io/badge/Automation-Autopy-purple)
![Platform](https://img.shields.io/badge/OS-Windows-red)
![Performance](https://img.shields.io/badge/Performance-30--35%20FPS-success)

---

**Control your computer mouse and keyboard using hand gestures and voice commands**.  
Built with **OpenCV**, **MediaPipe**, and **AI-powered hand tracking**, this project enables natural, touchless control of your computer — just with your hands.  

---

## ✨ Features  

- 🤖 **AI-Powered Hand Tracking** — Real-time 21-point landmark detection via **MediaPipe Hands**  
- 🖱️ **Full Mouse Control** — Move, click, double-click, right-click, and drag  
- 🧭 **Scroll Gestures** — Scroll up/down using open or closed hand  
- 🎙️ **Voice Typing Mode** — Activate speech-to-text by raising only the middle finger  
- 🪄 **Smooth & Stable Motion** — Built-in smoothing and gesture confidence filtering  
- ⚙️ **Configurable Active Region** — Define control area (`FRAME_R`) for precision  
- 🧠 **Multithreaded Design** — Voice input runs in parallel for zero lag   

---

## 🛠️ Installation  

### 1️⃣ Requirements  
- Python **3.8+**
- A **webcam**
- A **microphone** (for speech typing)


### 2️⃣ Install Dependencies  

```bash
pip install -r requirements.txt
```


### ▶️ Run the Application

```bash
python main.py
```

The camera window will appear — keep your hand inside the green control region and use gestures to control the mouse.

**Press ESC anytime to exit safely.**

---

## 🧠 Gesture Controls

|Gesture |	Action |	Description |
|--------|--------|-------------|
| 👆 | Index finger up |	Move Cursor	Move your hand to move the mouse |
| ✌️ | Index + Middle up |	Left Click	Hold steady to perform a left click |
| 🤟 | Index + Pinky up |	Double Click	Quick double tap gesture |
| 👍 | Thumb up only |	Right Click	Performs a right-click |
| 🖐️ | All fingers up	| Scroll Up	Scrolls upward |
| ✊ | All fingers down	Scroll Down |	Scrolls downward |
| 🖖 | All except thumb up |	Drag Mode	Click and hold for drag |
| 🖕 | Middle finger up only	| Voice Typing Mode	Enables speech-to-text typing

---

## ⚙️ Calibration & Tips


### Setting	Description

- 💡 Lighting: Bright, even lighting ensures best accuracy.
- 📏 Distance: Keep hand 1–2 ft from the webcam.
- 🟩 Active Region: Stay inside the green box for control.
- ⏳ Stability: Hold gestures steady for ~0.15s to trigger actions.
- 🛠️ Performance: Tune **SMOOTHING and FRAME_INSET_**... variables in the code to calibrate sensitivity.


### ⚡ Performance Optimization

- 🔒 Close other camera-using apps
- 💡 Ensure good lighting and contrast
- 🚀 Use a USB 3.0 webcam for higher FPS
- 🛠️ Adjust **SMOOTHING (default = 6)** to control responsiveness
- ⚡ Turn off debug overlays by setting **DEBUG_MODE = False** for maximum speed

---

## 🧩 Project Structure
```
📁 Hand Gesture AI Mouse
 ┣ 📜 main.py                 # Main program
 ┣ 📜 requirements.txt        # Dependencies
 ┗ 📜 README.md               # Documentation
```

---

## 🧰 Troubleshooting

### Issue	Solution

- 📷 Camera not detected: Ensure no other apps use it; try changing **cv2.VideoCapture(1)**
- 🗣️ Speech not working: Check mic permissions or install pyaudio / sounddevice
- 📉 Low FPS: Reduce resolution in code (CAM_WIDTH, CAM_HEIGHT)
- ✋ Gestures not detected: Improve lighting and background contrast
- 🧊 Program freezes: Disable speech mode or reduce speech phrase time limit

  ---
  
## 📘 Technical Summary

### Component	Details

- 🤖 AI Model: MediaPipe Hands (21 landmarks per hand)
- 🗺️ Cursor Mapping: Interpolation from camera to screen coordinates
- ⏳ Gesture Filtering: Time-based stability filter **(CLICK_CONFIDENCE_TIME)**
- 🗣️ Speech Engine: Google Speech-to-Text API
- 🚀 FPS: ~30–35 on standard webcam
- 🧵 Threading: Separate thread for speech typing
- 🛡️ Error Handling: Graceful fallback and recovery from errors

### ⌨️ Keyboard Shortcuts

| Key |	Function |
|-----|----------|
| ESC	| Exit the application |
| d (optional) |	Toggle debug visuals (landmarks, info) |

## 🧑‍💻 Author
- **[Afsal Rahiman T](https://github.com/afsal1122)** - Project Creator & Maintainer

---

## 🙏 Acknowledgments

- **MediaPipe team** at Google for their incredible hand-tracking model.
- **OpenCV team** for the essential computer vision library.
- **Autopy developers** for a simple, cross-platform system control library.
- **SpeechRecognition library** for making voice control so accessible.

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests.
```
Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
```

---

## 📄 License

**This project is released for educational and personal use.**
**Feel free to modify and expand it for research or development purposes.**

---

**⭐ If you find this project useful, consider starring it on GitHub! ⭐**