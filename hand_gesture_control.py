import cv2
import numpy as np
from flask import Blueprint, render_template, Response, request, jsonify
import mediapipe as mp
import pyautogui
from collections import deque
import random
import time

hand_gesture_control_bp = Blueprint('hand_gesture_control', __name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def generate_frames_control_mouse():
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()
    cursor_positions = deque(maxlen=10)
    alpha = 0.2
    ema_x, ema_y = None, None

    def calculate_distance(point1, point2):
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                x = int(index_finger_tip.x * w)
                y = int(index_finger_tip.y * h)
                screen_x = int(index_finger_tip.x * screen_width)
                screen_y = int(index_finger_tip.y * screen_height)

                if ema_x is None:
                    ema_x, ema_y = screen_x, screen_y
                else:
                    ema_x = alpha * screen_x + (1 - alpha) * ema_x
                    ema_y = alpha * screen_y + (1 - alpha) * ema_y

                pyautogui.moveTo(int(ema_x), int(ema_y))

                distance_index_thumb = calculate_distance(index_finger_tip, thumb_tip)
                distance_middle_thumb = calculate_distance(middle_finger_tip, thumb_tip)

                if distance_index_thumb < 0.05:
                    pyautogui.click(button='left')

                if distance_middle_thumb < 0.05:
                    pyautogui.click(button='right')

                if distance_index_thumb > 0.1:
                    pyautogui.scroll(int((distance_index_thumb - 0.1) * 500))

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def generate_frames_fruit_ninja():
    cap = cv2.VideoCapture(0)
    circles = deque(maxlen=10)
    particles = deque(maxlen=100)
    score = 0
    start_time = time.time()
    game_over = False

    def create_circle():
        return {
            'x': random.randint(50, 590),
            'y': random.randint(50, 430),
            'radius': 30,
            'color': (0, 255, 0),
            'type': 'fruit'
        }

    def create_bomb():
        return {
            'x': random.randint(50, 590),
            'y': random.randint(50, 430),
            'radius': 30,
            'color': (0, 0, 255),
            'type': 'bomb'
        }

    def create_particles(x, y, color, count=10, velocity_range=3):
        particles = []
        for _ in range(count):
            particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-velocity_range, velocity_range),
                'vy': random.uniform(-velocity_range, velocity_range),
                'color': color,
                'lifetime': random.uniform(0.5, 1.0)
            })
        return particles

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        for particle in list(particles):
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['lifetime'] -= 0.05
            if particle['lifetime'] <= 0:
                particles.remove(particle)
            else:
                cv2.circle(frame, (int(particle['x']), int(particle['y'])), 5, particle['color'], -1)

        for circle in list(circles):
            cv2.circle(frame, (circle['x'], circle['y']), circle['radius'], circle['color'], -1)

        if result.multi_hand_landmarks and not game_over:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_x = int(index_finger_tip.x * w)
                index_finger_tip_y = int(index_finger_tip.y * h)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for circle in list(circles):
                    dist = np.linalg.norm(np.array([circle['x'], circle['y']]) - np.array([index_finger_tip_x, index_finger_tip_y]))
                    if dist < circle['radius']:
                        if circle['type'] == 'bomb':
                            game_over = True
                            particles.extend(create_particles(circle['x'], circle['y'], circle['color'], count=20, velocity_range=5))
                        else:
                            circles.remove(circle)
                            particles.extend(create_particles(circle['x'], circle['y'], circle['color']))
                            score += 1
                        break

        if time.time() - start_time > 2 and not game_over:
            if random.random() < 0.1:
                circles.append(create_bomb())
            else:
                circles.append(create_circle())
            start_time = time.time()

        if game_over:
            cv2.putText(frame, 'Game Over', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@hand_gesture_control_bp.route('/video_feed')
def video_feed():
    program = request.args.get('program')
    if program == 'control_mouse':
        return Response(generate_frames_control_mouse(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif program == 'fruit_ninja':
        return Response(generate_frames_fruit_ninja(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'error': 'Invalid program specified.'}), 400

@hand_gesture_control_bp.route('/start_gesture_program', methods=['GET'])
def start_gesture_program():
    program = request.args.get('program')
    if program in ['control_mouse', 'fruit_ninja']:
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid program specified.'})

@hand_gesture_control_bp.route('/')
def index():
    return render_template('hand_gesture_page.html')
