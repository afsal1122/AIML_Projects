import cv2
import numpy as np
import mediapipe as mp
import autopy
import pyautogui
import time
import math
import threading
import queue
import traceback

# Try to import speech recognition and audio libraries
# Fallback from pyaudio to sounddevice if needed
try:
    import speech_recognition as sr
    try:
        import pyaudio
        AUDIO_LIB = "pyaudio"
    except ImportError:
        print("pyaudio not found, falling back to sounddevice.")
        try:
            import sounddevice
            AUDIO_LIB = "sounddevice"
        except ImportError:
            print("Neither pyaudio nor sounddevice found. Speech recognition will be disabled.")
            AUDIO_LIB = None
except ImportError:
    print("speech_recognition library not found. Speech recognition will be disabled.")
    sr = None
    AUDIO_LIB = None

# --- Global Configuration (UPDATED) ---

CAM_WIDTH, CAM_HEIGHT = 1280, 720 

# The green "active box" will be drawn using these margins.

FRAME_INSET_TOP = 25   
# *** MODIFIED: Lowered the bottom margin to increase length ***
FRAME_INSET_BOTTOM = 250  

FRAME_INSET_LEFT = 150    # Pixels from left 
FRAME_INSET_RIGHT = 150   # Pixels from right 
# You can increase this more (e.g., 10) for even smoother, but "heavier" movement.
SMOOTHING = 7
CLICK_CONFIDENCE_TIME = 0.2    # Time (sec) gesture must be held for a click
DEBUG_MODE = True               # Show debug overlays (landmarks, boxes, text)
# ---------------------------------------------------

class GestureRecognizer:
    """
    Handles hand tracking, landmark detection, and basic gesture classification
    (e.g., "fingers up").
    """
    def __init__(self, max_hands=1, min_detect_conf=0.7, min_track_conf=0.5):
        """
        Initializes the MediaPipe Hands model.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=min_track_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]  # Landmark IDs for fingertips
        self.landmarks = None
        self.results = None
        self.handedness = None

    def find_hands(self, img, draw=True):
        """
        Detects hands in an image frame.
        :param img: BGR image from OpenCV
        :param draw: Whether to draw landmarks and connections
        :return: The image, possibly with drawings
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        self.handedness = None

        if self.results.multi_hand_landmarks:
            hand_lms = self.results.multi_hand_landmarks[0]
            self.landmarks = hand_lms.landmark
            
            # Get handedness (Left or Right)
            if self.results.multi_handedness:
                self.handedness = self.results.multi_handedness[0].classification[0].label

            if draw:
                self.mp_draw.draw_landmarks(img, hand_lms,
                                            self.mp_hands.HAND_CONNECTIONS)
        else:
            self.landmarks = None
            
        return img

    def get_landmark_list(self, img_shape):
        """
        Extracts landmark coordinates into a list and calculates a bounding box.
        """
        lm_list = []
        bbox = []
        if self.landmarks:
            h, w, _ = img_shape
            x_all, y_all = [], []
            
            for id, lm in enumerate(self.landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                x_all.append(cx)
                y_all.append(cy)

            xmin, xmax = min(x_all), max(x_all)
            ymin, ymax = min(y_all), max(y_all)
            bbox = [xmin, ymin, xmax, ymax]
            
        return lm_list, bbox

    def fingers_up(self, lm_list):
        """
        Determines which fingers are extended based on landmark positions.
        *** This logic is updated to be more robust. ***
        """
        if not lm_list:
            return []
            
        fingers = []

        # *** ROBUST THUMB FIX (Distance Based) ***
        # Calculate distance between Thumb Tip (4) and Middle Finger MCP (9)
        x4, y4 = lm_list[4][1], lm_list[4][2]
        x9, y9 = lm_list[9][1], lm_list[9][2]
        thumb_dist = math.hypot(x4 - x9, y4 - y9)
        
        # Calculate scale of hand (Wrist (0) to Middle Finger MCP (9))
        x0, y0 = lm_list[0][1], lm_list[0][2]
        scale_dist = math.hypot(x0 - x9, y0 - y9)
        
        # Threshold: If thumb tip is far enough from the center of the palm, it's open.
        if thumb_dist > 0.4 * scale_dist:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 Fingers: Check if tip Y is above (lower value) the joint 2 steps down
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def find_distance(self, p1_id, p2_id, lm_list):
        """
        Calculates the distance between two landmarks.
        """
        x1, y1 = lm_list[p1_id][1], lm_list[p1_id][2]
        x2, y2 = lm_list[p2_id][1], lm_list[p2_id][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        return length, (cx, cy)


class GestureController:
    """
    Maps recognized gestures to system actions (mouse control, clicks, speech).
    """
    def __init__(self, screen_w, screen_h, insets, smoothing):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.inset_top, self.inset_bottom, self.inset_left, self.inset_right = insets
        self.smoothing = smoothing
        
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        
        self.is_dragging = False
        self.last_gesture = None
        self.gesture_start_time = 0
        self.action_triggered = False

    def process_gesture(self, gesture, lm_list, cam_w, cam_h, img):
        """
        Main logic function to interpret gestures and perform actions.
        """
        
        if not gesture:
            # Release drag if hand is lost
            if self.is_dragging:
                autopy.mouse.toggle(down=False)
                self.is_dragging = False
                print("Drag Released (Hand Lost)")
            self.last_gesture = None
            self.action_triggered = False
            return None

        # --- Get coordinates for cursor ---
        index_tip = lm_list[8][1], lm_list[8][2]
        cx, cy = index_tip

        # --- 1. Map Coordinates from Camera to Screen ---
        # Uses the four new inset values
        x_mapped = np.interp(cx, [self.inset_left, cam_w - self.inset_right], [0, self.screen_w])
        y_mapped = np.interp(cy, [self.inset_top, cam_h - self.inset_bottom], [0, self.screen_h])
        
        # --- 2. Smooth Cursor Motion ---
        self.curr_x = self.prev_x + (x_mapped - self.prev_x) / self.smoothing
        self.curr_y = self.prev_y + (y_mapped - self.prev_y) / self.smoothing

        # Hand-left now moves mouse-left
        move_x = self.curr_x
        move_y = self.curr_y
        
        # Clamp values to screen boundaries to prevent autopy error
        move_x = max(0, min(move_x, self.screen_w - 1))
        move_y = max(0, min(move_y, self.screen_h - 1))

        # --- 3. Gesture Stability Filter ---
        if gesture != self.last_gesture:
            self.last_gesture = gesture
            self.gesture_start_time = time.time()
            self.action_triggered = False # Reset trigger for new gesture
            if self.is_dragging and gesture != [0, 1, 1, 1, 1]:
                 autopy.mouse.toggle(down=False)
                 self.is_dragging = False
                 print("Drag Released (Gesture Change)")
            return None # Wait for confidence

        if time.time() - self.gesture_start_time < CLICK_CONFIDENCE_TIME:
            # Allow movement update while waiting for confidence
            if gesture == [0, 1, 0, 0, 0] or gesture == [1, 1, 0, 0, 0] or gesture == [0, 1, 1, 1, 1]:
                 autopy.mouse.move(move_x, move_y)
                 self.prev_x, self.prev_y = self.curr_x, self.curr_y
            return "..." # Waiting for confidence

        # --- 4. Perform Actions Based on Confident Gesture ---
        self.gesture_start_time = time.time()
        action = None

        # GESTURE: Move Cursor (Index up OR Index + Thumb up)
        if gesture == [0, 1, 0, 0, 0] or gesture == [1, 1, 0, 0, 0]:
            action = "Move"
            autopy.mouse.move(move_x, move_y)
            if DEBUG_MODE:
                cv2.circle(img, index_tip, 10, (0, 255, 0), cv2.FILLED)

        # GESTURE: Drag (All fingers up except thumb)
        elif gesture == [0, 1, 1, 1, 1]:
            action = "Drag"
            if not self.is_dragging:
                autopy.mouse.toggle(down=True)
                self.is_dragging = True
                print("Drag Started")
            autopy.mouse.move(move_x, move_y)
            if DEBUG_MODE:
                cv2.circle(img, index_tip, 10, (0, 0, 255), cv2.FILLED)
        
        # --- Other gestures (clicks, scrolls) only if NOT dragging ---
        elif not self.is_dragging:
            
            # GESTURE: Left Click (Index + Middle up)
            if gesture == [0, 1, 1, 0, 0]:
                if not self.action_triggered:
                    action = "Left Click"
                    print(action)
                    autopy.mouse.click()
                    self.action_triggered = True

            # GESTURE: Double Click (Index + Pinky up)
            elif gesture == [0, 1, 0, 0, 1]:
                if not self.action_triggered:
                    action = "Double Click"
                    print(action)
                    autopy.mouse.click(delay=0.01)
                    autopy.mouse.click(delay=0.01)
                    self.action_triggered = True
            
            # GESTURE: Right Click (Thumb up only)
            elif gesture == [1, 0, 0, 0, 0]:
                if not self.action_triggered:
                    action = "Right Click"
                    print(action)
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                    self.action_triggered = True
            
            # GESTURE: Scroll Up (All fingers up)
            elif gesture == [1, 1, 1, 1, 1]:
                action = "Scroll Up"
                print(action)
                pyautogui.scroll(100) # FIXED: Use pyautogui

            # GESTURE: Scroll Down (All fingers down / fist)
            elif gesture == [0, 0, 0, 0, 0]:
                action = "Scroll Down"
                print(action)
                pyautogui.scroll(-100) # FIXED: Use pyautogui

            # GESTURE: Speech Typing (Middle finger up only)
            elif gesture == [0, 0, 1, 0, 0]:
                action = "Speech Mode"
                # Action is handled in main loop to manage thread

        # Update previous location
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        
        return action


def recognize_speech_thread(speech_queue, stop_event):
    """
    Thread function for non-blocking speech recognition.
    """
    if sr is None or AUDIO_LIB is None:
        print("Speech recognition disabled in thread.")
        return

    recognizer = sr.Recognizer()
    
    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"FATAL: No microphone found or error initializing: {e}")
        speech_queue.put("ERROR: No Mic")
        return

    print("Speech thread started. Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Calibration complete. Listening...")
    
    while not stop_event.is_set():
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=1.5, phrase_time_limit=5)
            
            text = recognizer.recognize_google(audio)
            
            if text:
                print(f"Recognized: {text}")
                speech_queue.put(text + " ")

        except sr.UnknownValueError:
            pass # Silence, ignore
        except sr.WaitTimeoutError:
            pass # No speech, loop again
        except sr.RequestError as e:
            print(f"Speech API Error: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Speech Thread Error: {e}")
            time.sleep(1)
            
    print("Speech thread stopping.")


def main():
    """
    Main function to run the AI Gesture Mouse application.
    """
    p_time = 0
    
    # --- Initialize Camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    actual_cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Initialize System and Classes ---
    screen_w, screen_h = autopy.screen.size()
    recognizer = GestureRecognizer()
    
    # *** MODIFIED *** Pass asymmetrical insets to the controller
    insets = (FRAME_INSET_TOP, FRAME_INSET_BOTTOM, FRAME_INSET_LEFT, FRAME_INSET_RIGHT)
    controller = GestureController(screen_w, screen_h, insets, SMOOTHING)

    # --- Speech Recognition Thread Setup ---
    speech_queue = queue.Queue()
    speech_stop_event = threading.Event()
    speech_thread = None
    is_speech_mode = False
    last_speech_toggle_time = 0
    
    print("AI Gesture Mouse Control starting...")
    print(f"Screen Size: {screen_w} x {screen_h}")
    print(f"Requested Camera: {CAM_WIDTH} x {CAM_HEIGHT}")
    print(f"ACTUAL Camera:   {actual_cam_w} x {actual_cam_h}")
    print(f"Active Area Insets (Top, Bottom, Left, Right): {insets}")
    print("--- Mapping ACTIVE AREA (green box) to FULL screen ---")
    print("Press 'ESC' to quit.")

    # --- Main Loop ---
    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Failed to read frame from camera.")
                break
            
            img = cv2.flip(img, 1)

            # --- 1. Hand Tracking ---
            img = recognizer.find_hands(img, draw=DEBUG_MODE)
            lm_list, bbox = recognizer.get_landmark_list(img.shape)
            
            current_gesture = None
            action = None

            if lm_list:
                # --- 2. Gesture Recognition ---
                current_gesture = recognizer.fingers_up(lm_list)
                
                # --- 3. Gesture Control ---
                action = controller.process_gesture(
                    current_gesture, lm_list, actual_cam_w, actual_cam_h, img
                )

            # --- 4. Speech Mode Logic (Toggle) ---
            if current_gesture == [0, 0, 1, 0, 0] and action == "Speech Mode":
                current_time = time.time()
                if (current_time - last_speech_toggle_time) > 2.0: # 2 second debounce
                    last_speech_toggle_time = current_time
                    
                    if not is_speech_mode:
                        # Turn ON
                        if sr is not None:
                            print("STARTING SPEECH MODE")
                            is_speech_mode = True
                            speech_stop_event.clear()
                            speech_thread = threading.Thread(
                                target=recognize_speech_thread,
                                args=(speech_queue, speech_stop_event),
                                daemon=True
                            )
                            speech_thread.start()
                    else:
                        # Turn OFF
                        print("STOPPING SPEECH MODE")
                        is_speech_mode = False
                        speech_stop_event.set()
                        if speech_thread:
                            speech_thread.join(timeout=1)
                        speech_thread = None
                        while not speech_queue.empty():
                            speech_queue.get()

            # --- 5. Process Speech Queue ---
            try:
                text_to_type = speech_queue.get_nowait()
                if "ERROR:" in text_to_type:
                    print(f"Speech Thread Error: {text_to_type}")
                    is_speech_mode = False
                    speech_stop_event.set()
                else:
                    print(f"Typing: {text_to_type}")
                    autopy.key.type_string(text_to_type)
            except queue.Empty:
                pass # No text to type

            # --- 6. Debug Display ---
            if DEBUG_MODE:
                # Draw the new ASYMMETRICAL green box
                cv2.rectangle(img, (FRAME_INSET_LEFT, FRAME_INSET_TOP), 
                              (actual_cam_w - FRAME_INSET_RIGHT - 1, actual_cam_h - FRAME_INSET_BOTTOM - 1), 
                              (0, 255, 0), 2)
                
                # Draw FPS
                c_time = time.time()
                fps = 1 / (c_time - p_time)
                p_time = c_time
                cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                
                # Draw Gesture
                gesture_str = " ".join(map(str, current_gesture)) if current_gesture else "No Hand"
                cv2.putText(img, f"Gesture: {gesture_str}", (10, 70), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                
                if action:
                    cv2.putText(img, f"Action: {action}", (10, 110), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                
                if is_speech_mode:
                    cv2.putText(img, "Speech Mode: ON", (10, 150), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # --- 7. Show Image ---
            cv2.imshow("AI Gesture Mouse", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
                
    except Exception as e:
        print("An error occurred in the main loop:")
        print(traceback.format_exc())
        
    finally:
        # --- 8. Cleanup ---
        print("\nShutting down...")
        if speech_thread and speech_thread.is_alive():
            speech_stop_event.set()
            speech_thread.join()
        
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()