import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np


class ElbowGestureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_action = time.time()
        self.cooldown = 2.0  # 2 seconds cooldown
        self.mode = "SLIDE_MODE"
        self.last_mode_switch = time.time()
        self.mode_switch_cooldown = 1.0
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1
        
    def calculate_distance_to_shoulder_line(self, p1, p2, p3):
        """Calculate perpendicular distance from point p3 to line formed by p1-p2"""
        # Convert to numpy arrays
        p1 = np.array([p1.x, p1.y])
        p2 = np.array([p2.x, p2.y])
        p3 = np.array([p3.x, p3.y])
        
        # Calculate distance
        return np.abs(np.cross(p2-p1, p3-p1) / np.linalg.norm(p2-p1))
        
    def control_mouse(self, hand_landmark, frame_shape):
        try:
            h, w, _ = frame_shape
            x = hand_landmark.x * w
            y = hand_landmark.y * h
            
            # Convert to screen coordinates with bounds checking
            screen_w, screen_h = pyautogui.size()
            screen_x = min(max(0, np.interp(x, [0, w], [0, screen_w])), screen_w-1)
            screen_y = min(max(0, np.interp(y, [0, h], [0, screen_h])), screen_h-1)
            
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)
        except Exception as e:
            print(f"Mouse control error: {e}")

    def switch_mode(self):
        try:
            if self.mode == "MOUSE_MODE":
                # Reset mouse position when switching back to slide mode
                screen_w, screen_h = pyautogui.size()
                pyautogui.moveTo(screen_w//2, screen_h//2)
            self.mode = "MOUSE_MODE" if self.mode == "SLIDE_MODE" else "SLIDE_MODE"
            self.last_mode_switch = time.time()
        except Exception as e:
            print(f"Mode switch error: {e}")

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate distances
            left_dist = self.calculate_distance_to_shoulder_line(
                left_shoulder, right_shoulder, left_elbow)
            right_dist = self.calculate_distance_to_shoulder_line(
                left_shoulder, right_shoulder, right_elbow)
            
            # Mode switching logic
            current_time = time.time()
            if (left_dist < 0.15 and right_dist < 0.15 and 
                current_time - self.last_mode_switch > self.mode_switch_cooldown):
                self.switch_mode()
            
            # Mode-specific controls with error handling
            try:
                if self.mode == "SLIDE_MODE":
                    threshold = 0.15  # Updated threshold
                    if current_time - self.last_action >= self.cooldown:
                        if left_dist > threshold and right_dist <= threshold:
                            pyautogui.press('left')
                            self.last_action = current_time
                        elif right_dist > threshold and left_dist <= threshold:
                            pyautogui.press('right')
                            self.last_action = current_time
                else:  # MOUSE_MODE
                    self.control_mouse(right_wrist, frame.shape)
            except Exception as e:
                print(f"Control error: {e}")
            
            # Visualization
            h, w, c = frame.shape
            for point in [left_shoulder, right_shoulder, left_elbow, right_elbow]:
                cx, cy = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            
            cv2.line(frame, 
                    (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                    (int(right_shoulder.x * w), int(right_shoulder.y * h)),
                    (255, 0, 0), 2)
            
            # Display mode and distances
            cv2.putText(frame, f"Mode: {self.mode}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"L: {left_dist:.2f} R: {right_dist:.2f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = ElbowGestureDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = detector.process_frame(frame)
        cv2.imshow('Elbow Distance Control', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()