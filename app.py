from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import speech_recognition as sr
import numpy as np

app = Flask(__name__)

video_capture = None
recording = False
face_centered = 0
motion_frames = 0
start_time = 0
frame_count = 0

def record_loop(timeout=30):
    global video_capture, recording, face_centered, motion_frames, start_time, frame_count

    face_centered = 0
    motion_frames = 0
    frame_count = 0
    start_time = time.time()
    recording = True

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    prev_gray = None

    while recording and (time.time() - start_time) < timeout:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face Detection + Center Check (proportional)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            frame_width, frame_height = gray.shape[1], gray.shape[0]
            if abs(cx - frame_width // 2) < frame_width * 0.1 and abs(cy - frame_height // 2) < frame_height * 0.1:
                face_centered += 1
                break

        # Motion Detection using optical flow
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitude = np.mean(mag)
            if motion_magnitude > 1.2:
                motion_frames += 1

        prev_gray = gray

    recording = False

def gen_frames():
    global video_capture
    while video_capture and video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video')
def start_video():
    global video_capture, recording
    if recording:
        return jsonify({'status': 'Already recording'})
    video_capture = cv2.VideoCapture(0)
    threading.Thread(target=record_loop, daemon=True).start()
    return jsonify({'status': 'Recording started'})

@app.route('/stop_video')
def stop_video():
    global recording, video_capture
    recording = False
    if video_capture:
        video_capture.release()
        video_capture = None
    return jsonify({'status': 'Recording stopped'})

@app.route('/analyze')
def analyze():
    global face_centered, motion_frames, frame_count

    # Ratio-based scoring
    center_ratio = face_centered / frame_count if frame_count else 0
    motion_ratio = motion_frames / frame_count if frame_count else 0

    center_score = 20 if center_ratio >= 0.6 else 10 if center_ratio >= 0.3 else 0
    motion_score = 20 if motion_ratio >= 0.5 else 10 if motion_ratio >= 0.2 else 0

    # Speech recognition
    recognizer = sr.Recognizer()
    speech_score = 0
    spoken_text = ""

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for speech...")
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=15)
            spoken_text = recognizer.recognize_google(audio, language='en-US')
            word_count = len(spoken_text.split())
            speech_score = 30 if word_count >= 10 else 15 if word_count >= 5 else 0
    except sr.UnknownValueError:
        spoken_text = "Speech not recognized clearly."
    except sr.RequestError:
        spoken_text = "Speech API request failed."
    except Exception as e:
        spoken_text = f"Speech error: {str(e)}"

    total_score = center_score + motion_score + speech_score

    return jsonify({
        'total_score': total_score,
        'face_score': center_score,
        'motion_score': motion_score,
        'speech_score': speech_score,
        'spoken_text': spoken_text,
        'face_centered_frames': face_centered,
        'motion_detected_frames': motion_frames
    })

if __name__ == '__main__':
    app.run(debug=True)
