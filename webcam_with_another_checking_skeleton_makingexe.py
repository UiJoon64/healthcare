from flask import Flask, render_template, Response, jsonify
import os
import torch
import cv2
import time
import numpy as np
from torchvision import transforms
from collections import deque
from utils.datasets import letterbox
#from utils.torch_utils import select_device
from models.experimental import attempt_load
from tensorflow.keras.models import load_model
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

joint_angle_queue = deque(maxlen=5)

action_type = {0:"낙상",1:"일상",2:"눕기"}

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)  # 웹캠 비디오 캡처 객체 생성

sequential_model = load_model('./latest_sequential_model.h5')

model_name = "yolov7-w6-pose.pt"
device = torch.device('cpu')
model = attempt_load(model_name, map_location=device)
_ = model.eval()
names = model.module.names if hasattr(model, 'module') else model.names

current_angles = {'elbow': 0, 'shoulder': 0, 'hip': 0, 'knee': 0}
action_result = ""

pairs = [(0, 1), (0, 2), (1, 3), (2, 4),   # 머리-목-어깨 
         (5, 6), (5, 7), (7, 9), (6, 8),(8,10), # 왼쪽 손목-팔꿈치-어깨 / 오른쪽 손목-팔꿈치-어깨 
         (11,13),(13,15),(12,14),(14,16)]

fall_count = 0  # "낙상" 판정 카운트
fall_threshold = 2  # "낙상" 판정 임계값

# 관절 각도 계산 함수
def calculate_angle(a_pt,b_pt,c_pt):
    a_pt=np.array(a_pt)
    b_pt=np.array(b_pt)
    c_pt=np.array(c_pt)

    radians= np.arctan2(c_pt[1]-b_pt[1],c_pt[0]-b_pt[0]) - np.arctan2(a_pt[1]-b_pt[1],a_pt[0]-b_pt[0])
    angle=np.abs(radians*180.00/np.pi)

    if angle >180.00:
        angle=360-angle

    return angle 

# 시작화면 - 로그인/회원가입
@app.route('/', methods=['GET', 'POST'])
def main():
    error = None

    return render_template('main.html', error=error)


# 회원가입 화면
@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None

    return render_template('register.html', error=error)

@app.route('/video_image')
def index():
     data={} # 빈 딕셔너리 생성 
     return render_template('video_image.html',data=data)

def generate_frames():
    global current_angles
    global action_result
    
    frame_width = int(video_capture.get(3))  #get video frame width
    frame_height = int(video_capture.get(4)) #get video frame height


    angles_to_calculate=[(9 ,7 ,5),(7 ,5 ,11),(5 ,11 ,13),(11 ,13 ,15)]
    angle_dict = {angle: [] for angle in range(len(angles_to_calculate))}
    current_angles = {'elbow': 0, 'shoulder': 0, 'hip': 0, 'knee': 0}
    joint_names=['elbow','shoulder','hip','knee']

    while True :
        ret, frame=video_capture.read()
        if not ret :
            break 
        
        orig_image = frame #store frame
        image=cv2.cvtColor(orig_image,cv2.COLOR_BGR2RGB) #convert frame to RGB 
        image = letterbox(image, (frame_width), stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        image = image.to(device)  #convert image data to device
        image = image.float() #convert image to float precision (cpu)
        start_time = time.time() #start time for fps calculation

        with torch.no_grad() :
            output_data,_= model(image)
        
        output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
        output = output_to_keypoint(output_data)

        im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
        im0 = im0.cpu().numpy().astype(np.uint8)
        
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh


        for i, pose in enumerate(output_data):  # detections per image
        
            if len(output_data):  #check if no pose
                for c in pose[:, 5].unique(): # Print results
                    n = (pose[:, 5] == c).sum()  # detections per class
                    print("No of Objects in Current Frame : {}".format(n))
                
                max_conf_index = None
                max_conf = -1
                
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                    if conf > max_conf:
                        max_conf = conf
                        max_conf_index = det_index

                if max_conf_index is not None:
                    det_index = max_conf_index 
                    c = int(cls)  # integer class
                    kpts = pose[det_index, 6:]

                    for i in pairs:
                        partA = i[0]
                        partB = i[1]
                        if kpts[partA*3+2]>0 and kpts[partB*3+2]>0:
                            cv2.line(im0,(int(kpts[partA*3]),int(kpts[partA*3+1])),(int(kpts[partB*3]),int(kpts[partB*3+1])),(255,255,),5)

                    for angle_idx, (index_a,index_b,index_c) in enumerate(angles_to_calculate):
                        try:
                            a_pt = kpts[index_a*3:index_a*3+2]
                            b_pt = kpts[index_b*3:index_b*3+2]
                            c_pt = kpts[index_c*3:index_c*3+2]

                            angle_value = calculate_angle(a_pt,b_pt,c_pt)

                            cv2.putText(im0,str(angle_value),(int(b_pt[0]),int(b_pt[1])),cv2.FONT_HERSHEY_SIMPLEX ,  
                                        0.5,(255),2,cv2.LINE_AA)

                            current_angles[joint_names[angle_idx]] = angle_value


                        except Exception as e:
                            print(f"Error occurred: {str(e)}")
                    print(current_angles)

                joint_angle_queue.append(current_angles)

                # if len(joint_angle_queue) == 1: # queue가 비어있는 상태에서 첫 번째 요소가 추가될 때
                #     start_time_lstm = time.time() # 시작 시간 기록

                
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
                    
                    #end_time_lstm = time.time() # <- Add this line
                    #print(f"Time taken for prediction: {end_time_lstm - start_time_lstm} seconds")

                    if predicted_action == 0:   # "낙상"일 경우 카운트 증가
                        fall_count += 1
                        if fall_count >= fall_threshold:   # 카운트가 임계값 이상일 경우 "낙상" 확정 
                            action_result = action_type[predicted_action]
                            print("Fall detected!")
                        else:
                            #action_result = "Pending"
                            print("Pending...")
                            
                    else:   # 다른 행동일 경우 카운트 초기화 및 해당 행동 결과 저장 
                        fall_count = 0  
                        action_result = action_type[predicted_action]

                    #action_result = action_type[predicted_action]

                # 프레임을 JPEG 이미지로 인코딩
                ret, buffer = cv2.imencode('.jpg', im0)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/joint_angles')
def joint_angles():
    return jsonify(current_angles)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action_result')  # 엔드포인트 추가
def get_action_result():
    return jsonify({"result": action_result})

if __name__ == '__main__':
    app.run(debug=True)
