# 🖐️ Hand Gesture AI Mouse Control System  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-yellow)
![Autopy](https://img.shields.io/badge/Automation-Autopy-purple)
![Platform](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Performance](https://img.shields.io/badge/Performance-30--35%20FPS-success)

---

> **Control your computer mouse and keyboard using hand gestures and voice commands**.  
> Built with **OpenCV**, **MediaPipe**, and **AI-powered hand tracking**, this project enables natural, touchless control of your computer — just with your hands.  

---

## ✨ Features  

- 🤖 **AI-Powered Hand Tracking** — Real-time 21-point landmark detection via **MediaPipe Hands**  
- 🖱️ **Full Mouse Control** — Move, click, double-click, right-click, and drag  
- 🧭 **Scroll Gestures** — Scroll up/down using open or closed hand  
- 🎙️ **Voice Typing Mode** — Activate speech-to-text by raising only the middle finger  
- 🪄 **Smooth & Stable Motion** — Built-in smoothing and gesture confidence filtering  
- ⚙️ **Configurable Active Region** — Define control area (`FRAME_R`) for precision  
- 🧠 **Multithreaded Design** — Voice input runs in parallel for zero lag  
- 🧩 **Cross-Platform** — Works seamlessly on Windows, Linux, and macOS  

---

## 🛠️ Installation  

### 1️⃣ Requirements  
- Python **3.8+**
- A **webcam**
- A **microphone** (for speech typing)

### 2️⃣ Install Dependencies  

```bash
pip install -r requirements.txt
If pyaudio installation fails (Windows users):

bash
Copy code
pip install pipwin
pipwin install pyaudio
or use fallback audio libraries:

bash
Copy code
pip install sounddevice soundfile
▶️ Run the Application
bash
Copy code
python hand_gesture_ai_mouse.py
The camera window will appear — keep your hand inside the green control region and use gestures to control the mouse.

Press ESC anytime to exit safely.

🧠 Gesture Controls
Gesture	Action	Description
👆 Index finger up	Move Cursor	Move your hand to move the mouse
✌️ Index + Middle up	Left Click	Hold steady to perform a left click
🤟 Index + Pinky up	Double Click	Quick double tap gesture
👍 Thumb up only	Right Click	Performs a right-click
🖐️ All fingers up	Scroll Up	Scrolls upward
✊ All fingers down	Scroll Down	Scrolls downward
🖖 All except thumb up	Drag Mode	Click and hold for drag
🖕 Middle finger up only	Voice Typing Mode	Enables speech-to-text typing

⚙️ Calibration & Tips
Setting	Description
Lighting	Bright, even lighting ensures best accuracy
Distance	Keep hand 1–2 ft from webcam
Active Region	Stay inside the green box for control
Stability	Hold gestures steady for ~0.15–0.3s
Performance	Set SMOOTHING and FRAME_R in code for sensitivity tuning

⚡ Performance Optimization
Close other camera-using apps

Ensure good lighting and contrast

Use USB 3.0 webcam for higher FPS

Adjust SMOOTHING (default = 5) to control responsiveness

Turn off debug overlays by setting DEBUG_MODE = False for maximum speed

🧩 Project Structure
bash
Copy code
📁 HandGestureAIMouse
 ┣ 📜 hand_gesture_ai_mouse.py      # Main program
 ┣ 📜 requirements.txt              # Dependencies
 ┗ 📜 README.md                     # Documentation

🧰 Troubleshooting
Issue	Solution
Camera not detected	Ensure no other apps use it; try changing cv2.VideoCapture(1)
Speech not working	Check mic permissions or install pyaudio / sounddevice
Low FPS	Reduce resolution in code (CAM_WIDTH, CAM_HEIGHT)
Gestures not detected	Improve lighting and background contrast
Program freezes	Disable speech mode or reduce speech phrase time limit

📘 Technical Summary
Component	Details
AI Model	MediaPipe Hands (21 landmarks per hand)
Cursor Mapping	Interpolation from camera to screen coordinates
Gesture Filtering	Time-based stability filter (CLICK_CONFIDENCE_TIME)
Speech Engine	Google Speech-to-Text API
FPS	~30–35 on standard webcam
Threading	Separate thread for speech typing
Error Handling	Graceful fallback and recovery from errors

⌨️ Keyboard Shortcuts
Key	Function
ESC	Exit the application
d (optional)	Toggle debug visuals (landmarks, info)

🧑‍💻 Author
Afsal Rahiman T

🪪 License
This project is released for educational and personal use.
Feel free to modify and expand it for research or development purposes.

⭐ If you find this project useful, consider starring it on GitHub! ⭐