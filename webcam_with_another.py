from flask import Flask, render_template, Response, jsonify
import os
import torch
import cv2
import time
import numpy as np
import base64
from PIL import Image
import io
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

joint_angle_queue = deque(maxlen=5)

action_type = {0:"낙상",1:"일상",2:"배회"}

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)  # 웹캠 비디오 캡처 객체 생성

sequential_model = load_model('./angle_classifying_model.h5')

model_name = "yolov7.pt"
model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
model.eval()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

current_angles = {'elbow': 0, 'shoulder': 0, 'hip': 0, 'knee': 0}
action_result = ""

# 바운딩박스 기준으로 이미지 crop하는 함수
def crop_objects(img, xyxy, crop_dir, p_stem, frame):
    crop_path = os.path.join(crop_dir, p_stem)
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    crop_img = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img_path = os.path.join(crop_path, f"frame_{frame}.jpg")
    cv2.imwrite(crop_img_path, crop_img)

# 바운딩 박스가 그려진 predict 이미지 가져와서, crop_objects 함수를 사용해 이미지 crop
def get_prediction(img_bytes, current_time):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = model(imgs, size=640)  # includes NMS
    detections = results.pred  # Get the list of detected objects

    # 바운딩 박스 좌표 추출
    if detections is not None and len(detections[0]) > 0:
        x1, y1, x2, y2 = detections[0][0][0].item(), detections[0][0][1].item(), detections[0][0][2].item(), detections[0][0][3].item()
        bounding_box = [x1, y1, x2, y2]
        confidence = detections[0][0][4].item()
        label = detections[0][0][5].item()

        # 사람일때만 이미지 crop
        if label == 0:
            crop_objects(imgs[0], bounding_box, 'C:\pukyung_202301\yolov7-object-cropping\special', 'detect_person', current_time)
    else:
        print("No person detected.")

    return results

# 관절 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# 관절 추정된 이미지로 저장하기
def save_image_with_landmarks(image, save_dir, current_time):
    save_path = os.path.join(save_dir, f"{str(current_time)}_capture_landmarks.jpg")
    cv2.imwrite(save_path, image)

@app.route('/')
def index():
    data = {}  # 빈 딕셔너리 생성
    return render_template('video_image.html', data=data)

def generate_frames():
    global action_result
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            # 웹캠에서 프레임 읽기
            success, frame = video_capture.read()
            if not success:
                break
            
            # Mediapipe용 input data로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Mediapipe로 처리
            result = pose.process(image)
            
            # 추가: MediaPipe 처리 결과 그려주기
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if result.pose_landmarks is not None:
                landmarks = result.pose_landmarks.landmark

                # Calculate angles
                angle_elbow = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])
                angle_shoulder = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y])
                angle_hip = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
                angle_knee = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])

                #print(f"Elbow angle: {angle_elbow} Shoulder angle: {angle_shoulder} Hip angle: {angle_hip} Knee angle: {angle_knee}")
                
                # 프레임에서 각도 계산하고 전역 변수로 저장
                current_angles["elbow"] = angle_elbow
                current_angles["shoulder"] = angle_shoulder
                current_angles["hip"] = angle_hip
                current_angles["knee"] = angle_knee
                
                joint_angle_queue.append(current_angles)
                
                
            if len(joint_angle_queue) == 5:
                # LSTM 모델 입력을 위한 관절 각도 준비
                joint_angle_array = []
                for angle in joint_angle_queue:
                    joint_angle_array.append([angle['elbow'], angle['shoulder'], angle['hip'], angle['knee']])
                joint_angle_array = np.array(joint_angle_array)
                joint_angle_array = joint_angle_array.reshape(1, 5, 4)  # LSTM 모델의 입력 형태에 맞게 조정

                # 행동 예측
                predicted_action = sequential_model.predict(joint_angle_array)
                predicted_action = int(np.argmax(predicted_action, axis=-1))  # 가장 높은 확률을 가진 행동의 인덱스를 얻음
                
                action_result = action_type[predicted_action]
                
                # 예측된 행동 출력
                #print(f"Predicted action: {action_result}")
                
            # 프레임을 JPEG 이미지로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            current_time = time.time()
            results = get_prediction(frame_bytes, current_time)

            # 스트리밍되는 프레임 반환
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            # 1초마다 캡처된 이미지 저장
            if int(time.time()) % 1 == 0:
                with open("static/capture.jpg", "wb") as f:
                    f.write(frame_bytes)
                #save_image_with_landmarks(image, 'static/saved_images', current_time)

@app.route('/joint_angles')
def joint_angles():
    return jsonify(current_angles)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/capture_image')
def capture_image():
    with open("static/capture.jpg", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_image}"

@app.route('/action_result')  # 엔드포인트 추가
def get_action_result():
    return jsonify({"result": action_result})


if __name__ == '__main__':
    app.run(debug=True)