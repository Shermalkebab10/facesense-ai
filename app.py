from flask import Flask, request, render_template, Response, session, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import threading
import base64
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'face_mask_emotion_secret_key'

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

mask_model = load_model('models/mask_model.h5')
emotion_model = load_model('models/emotion_model.h5')

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

MASK_LABELS = ['With Mask', 'Without Mask']
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def predict_face(face_img):
    mask_input = cv2.resize(face_img, (224, 224)) / 255.0
    mask_input = np.expand_dims(mask_input, axis=0)
    mask_pred = mask_model.predict(mask_input, verbose=0)[0][0]
    mask_label = MASK_LABELS[1] if mask_pred > 0.5 else MASK_LABELS[0]
    mask_conf = mask_pred if mask_pred > 0.5 else 1 - mask_pred

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    emo_input = cv2.resize(gray, (48, 48)) / 255.0
    emo_input = np.expand_dims(emo_input, axis=(0, -1))
    emo_pred = emotion_model.predict(emo_input, verbose=0)[0]
    emotion_label = EMOTION_LABELS[np.argmax(emo_pred)]
    emotion_conf = float(np.max(emo_pred))

    return mask_label, float(mask_conf), emotion_label, emotion_conf

def draw_on_frame(img, faces, results):
    for i, (x, y, w, h) in enumerate(faces):
        if i >= len(results):
            break
        r = results[i]
        color = (0, 200, 80) if r['mask'] == 'With Mask' else (0, 60, 220)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y - 52), (x + w, y), color, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, f"{r['mask']} {r['mask_conf']}",
                    (x+6, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
        cv2.putText(img, f"{r['emotion']} {r['emotion_conf']}",
                    (x+6, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
    return img

def generate_frames():
    frame_count = 0
    last_results = []
    last_faces = []

    while True:
        with camera_lock:
            cam = get_camera()
            success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 5 == 0:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            last_faces = faces if len(faces) else []
            last_results = []
            for (x, y, w, h) in last_faces:
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                mask_label, mask_conf, emotion_label, emotion_conf = predict_face(face_img)
                last_results.append({
                    'mask': mask_label,
                    'mask_conf': f"{mask_conf*100:.0f}%",
                    'emotion': emotion_label,
                    'emotion_conf': f"{emotion_conf*100:.0f}%"
                })

        for i, (x, y, w, h) in enumerate(last_faces):
            if i >= len(last_results):
                break
            r = last_results[i]
            color = (0, 200, 80) if r['mask'] == 'With Mask' else (0, 60, 220)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y-52), (x+w, y), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"{r['mask']} {r['mask_conf']}",
                        (x+6, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
            cv2.putText(frame, f"{r['emotion']} {r['emotion_conf']}",
                        (x+6, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)

        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def home():
    history = session.get('history', [])
    return render_template('index.html', history=history)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None
    return '', 204

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['history'] = []
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return render_template('index.html', error="Could not read image.",
                               history=session.get('history', []))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    results = []
    face_list = [(x, y, w, h) for (x, y, w, h) in faces]

    for (x, y, w, h) in face_list:
        face_img = img[y:y+h, x:x+w]
        mask_label, mask_conf, emotion_label, emotion_conf = predict_face(face_img)
        results.append({
            'mask': mask_label,
            'mask_conf': f"{mask_conf*100:.1f}%",
            'mask_conf_val': round(mask_conf * 100, 1),
            'emotion': emotion_label,
            'emotion_conf': f"{emotion_conf*100:.1f}%",
            'emotion_conf_val': round(emotion_conf * 100, 1)
        })

    img = draw_on_frame(img, face_list, results)
    result_filename = 'result_' + file.filename
    cv2.imwrite(os.path.join(RESULT_FOLDER, result_filename), img)

    # Save to history
    history = session.get('history', [])
    mask_count = sum(1 for r in results if r['mask'] == 'With Mask')
    history.insert(0, {
        'filename': result_filename,
        'time': datetime.now().strftime('%d %b %Y, %H:%M'),
        'faces': len(face_list),
        'mask_count': mask_count,
        'emotion': results[0]['emotion'] if results else 'N/A'
    })
    session['history'] = history[:10]

    return render_template('index.html',
                           result_image=result_filename,
                           results=results,
                           num_faces=len(face_list),
                           history=session.get('history', []))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
