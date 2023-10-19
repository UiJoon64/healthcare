👵 SENIOR_MOTION_DETECTION 👴
========================
노령층 이상행동 감지 및 보호자 알림 서비스


👨‍👨‍👧‍👧 멤버 구성
======================
팀장_전의준 : 🧭 제작 총괄

팀원_김민영 : 🔊 음성 데이터 분석 및 학습

팀원_김지아 : 🏃 이미지 및 영상 데이터 분석 및 학습

팀원_손수지 : 📊 웹 페이지 제작

팀원_윤지수 : 🏃 이미지 및 영상 데이터 분석 및 학습


⚙ 개발 환경
======================
PYTHON 3.9.16

CUDA 11.3

이미지데이터 : YOLOv7-tiny

영상데이터 : YOLOv7, media pipe

음성데이터 : webkitSpeechRecognition, BERT

웹 : 

(데이터 출처 : AI-Hub)


🌳 프로젝트 개요
=======================
65세 이상의 고령 인구가 증가함에 따라, 홀로 계시는 어르신 또한 많아지고 있다.

이 프로젝트는 혼자 계시는 어르신의 행동 특성과 발생 소리를 모니터링하여 

위험사항을 예측 및 감지하고 보호자에게 즉시 알리기 위해 제작되었다.

<전체 구성 요약>
![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/fc330b03-a08c-49c2-8c5f-b00b9718ae53)


⭐ 이미지데이터
=====================
AI-Hub에서 총 80가지의 일상생활 라벨링 데이터를 담은

**< 일상생활 이미지 데이터 >** 를 가져와, 가정에서 흔히 발생할 수 있는 **17가지의 상황만을 선별**하였다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/b4db6294-5ef4-4de8-ab6a-a93d9ebf09a1)


이미지들은 YOLOv7-tiny의 학습을 위해 이미지 크기를 1920x1080 사이즈에서 640x360의 사이즈로 조절하였다.

# val, train resize
import os
import glob
from PIL import Image
output_path =("C:/Users/Administrator/Desktop/일상생활 영상/train_resize/")

os4 = os.listdir("C:/Users/Administrator/Desktop/일상생활 영상/val")
for b in os4 :
    file_path4 = "C:/Users/Administrator/Desktop/일상생활 영상/val/{}/".format(b)
    os3 = os.listdir(file_path4)
    for a in os3 :
        file_path3 = file_path4 + "/{}/".format(a)
        os2 = os.listdir(file_path3)
        for j in os2 :
            file_path2 = file_path3 + "/{}/".format(j)
            os_list = os.listdir(file_path2)
            for i in os_list:
                # 타겟 폴더 정보
                file_path1 = file_path2 + "/{}/".format(i)
                list_images = os.listdir(file_path1)
                #print(list_images)
                
                *반복문
                for image in list_images[1:] :
                    # 이미지 가져와서 크기 조절
                    img = Image.open(file_path1+image)
                    print(image)
                    
                    (width, height) = (img.width//3, img.height//3)
                    resize_show = img.resize((width, height))
                    resize_show.save("C:/Users/Administrator/Desktop/일상생활 영상/val_resize/" +image)

이미지마다 각 행동의 바운딩박스 좌표가 들어있었고, 바운딩박스의 좌표 또한 이미지 크기 비율에 맞춰 정규화시켜주었다.

* YOLOv7-tiny를 선정한 이유는 YOLOv7-tiny가 객체 탐지의 가장 기본적이고 속도가 매우 빠른 알고리즘으로,

  프로젝트에 필요한 실시간 영상 처리에 적합하다고 생각했기에 선정하였다.
  
---------------------------------------------------------------------------------------------------------------------

이미지 사이즈와 바운딩박스 좌표 정규화가 끝난 5325개의 데이터들은 Batch Size=4, Epoch=100으로 YOLOv7-tiny를 통해 학습시켜

17가지의 행동 분류 결과를 텍스트로 추출한다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/029870ad-8f2c-4897-99e3-c0d5e8e26a85)


예를 들어, 말할까 ???????????????????



🌟 영상데이터
====================
AI-Hub에서 고령자 개인의 외형과 행위 특성(습관), 건강 상태, 생활 패턴 등을 담고 있는

< 이상행동 영상 데이터 > 를 가져와, 일상 생활에서 일어날 수 있다고 판단한 낙상/일상/배회 총 3가지 section의 데이터를 선별하였다.

이 프로젝트에서는 보통 재택 공간(실내)에서 일어나는 이상 상황을 감지하지만, 모델의 학습을 위해 실내와 실외 데이터 모두 사용하였다.


영상 데이터 중 낙상/일상/배회가 일어나는 순간인 2초 정도를 하이라이트로 가져와 영상의 전처리를 마쳤다.

-------------------------------------------------------------------------------------------------------------------------

전체적인 그림은

실시간 영상 > 관절 각도 추출 > 그래프 송출 및 이상행동 감지

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/a4b211e5-323b-4664-a21c-019a27ac6b87)

이때 관절 각도를 추출해 이상행동을 감지하는 방법은

낙상과 같은 이상자세, 보행과 같은 동적인 자세, 명상과 같은 정적인 자세의

관절 각도 차이가 확연하게 드러났기에 사용하게 되었다. 

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/856bf3b8-4846-42c7-a5a3-af01f2a14e6f)


**<LSTM>**

LSTM은 기존의 RNN을 보완해 장/단기 기억을 가능하게 설계한 신경망의 구조로 주로 시계열처리, 자연어처리에 사용된다.

이 프로젝트에서는 실시간 영상에서 관절 각도를 추출해내 그래프로 바로 보여주는데에 초점을 두었기에, LSTM을 사용하였다.

모델 학습 시에는 기존 375개의 데이터밖에 없었기에, 데이터를 증강하여 15000개의 데이터를 학습시켰다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/dc436f08-4b85-4053-a469-6fda449ab1df)


또한, 그래프를 그릴 때 최대한 끊김 없이 부드럽게 송출하기 위해 데이터를 1초에 한번씩 보여주는 것과 같이 짧게 끊는 것이 아닌,

3초에 한번씩 보여주도록 길게 끊어내 조금 더 부드럽게 만들었다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/5e54d7f7-00c5-4884-b2f2-85f07c2b319a)


**<YOLOv7, media pipe>**

media pipe를 이용해서 실시간 영상에서 자동으로 관절각도량을 추출해 그래프로 송출하려 했으나, 사람을 인식하는 정확도 면에서 문제가 발생했다.

따라서 YOLOv7을 활용해 사람만 먼저 인식한 후, 사람만 크롭한 영상을 media pipe에 넣어 관절 각도를 추출하는 형식으로 바꾸었다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/c4c1b048-6f8d-4a35-9331-1d50f0937d0b)


✨ 음성데이터
=====================
AI-Hub에서 "불이야!", " 도와주세요.."와 같은 위급상황 발생에 따른 다양한 상황과 환경별 음성 및 음향을 포함한 데이터를

실시간 영상에서 음성을 추출해내 그래프 아래에 보여주는데 활용하기 위해 가져왔다.

**<webkitSpeechRecognition>**

우선 음성인식 API로 크롬 브라우저를 사용해 쉽게 구현 가능한 webkitSpeechRecognition을 통해

음성을 문서로 변환해준다.

**<BERT>**

이후 텍스트 데이터로 사전 훈련된 높은 성능을 가진 언어 모델인 BERT 모델을 통해

문서로 변환된 음성의 상황을 인식해 위험 상황으로 인식되면 보호자에게 알린다.

이때 사용된 훈련 데이터는 19474개이며, Batch Size는 32로 총 15번 훈련시켰다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/fe15825e-19cb-42b8-949e-6e32df6f7e16)


💫 웹 서비스 구현
=====================
위 사용한 이미지 데이터, 영상 데이터, 음성 데이터를 모두 모아 한 페이지에 보여주기 위해 

파이썬 기반의 웹 프레임워크인 Flask를 사용해 웹 서비스를 만들었다.

Flask는 확장성이 뛰어나고 필요한 기능들을 추가할 수 있어 채택하게 되었다.


🤩 프로젝트 결과 및 실행
========================
![Uploading image.png…]()


