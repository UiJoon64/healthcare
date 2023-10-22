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

<사용 모델, api 등>
~~이미지데이터 : YOLOv7-tiny~~

영상데이터 : ~~YOLOv7~~, ~~media pipe~~, yolov7-w6-pose

음성데이터 : webkitSpeechRecognition, ~~BERT~~

로그인 DB : Firebase

(데이터 출처 : ~~AI-Hub~~)


🌳 프로젝트 개요
=======================
65세 이상의 고령 인구가 증가함에 따라, 홀로 계시는 어르신 또한 많아지고 있다.

이 프로젝트는 혼자 계시는 어르신의 행동 특성과 발생 소리를 모니터링하여 

위험사항을 예측 및 감지하고 보호자에게 즉시 알리기 위해 제작되었다.

<전체 구성 요약>
![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/fc330b03-a08c-49c2-8c5f-b00b9718ae53)

⭐ 이미지데이터
=====================
<details>
<summary>정확도 이슈로 인해 잠정 제외</summary>
<div markdown="1">




AI-Hub에서 총 80가지의 일상생활 라벨링 데이터를 담은

**< 일상생활 이미지 데이터 >** 를 가져와, 가정에서 흔히 발생할 수 있는 **17가지의 상황만을 선별**하였다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/b4db6294-5ef4-4de8-ab6a-a93d9ebf09a1)


이미지들은 YOLOv7-tiny의 학습을 위해 이미지 크기를 1920x1080 사이즈에서 640x360의 사이즈로 조절하였다.

이미지마다 각 행동의 바운딩박스 좌표가 들어있었고, 바운딩박스의 좌표 또한 이미지 크기 비율에 맞춰 정규화시켜주었다.

* YOLOv7-tiny를 선정한 이유는 YOLOv7-tiny가 객체 탐지의 가장 기본적이고 속도가 매우 빠른 알고리즘으로,

  프로젝트에 필요한 실시간 영상 처리에 적합하다고 생각했기에 선정하였다.
  
---------------------------------------------------------------------------------------------------------------------

이미지 사이즈와 바운딩박스 좌표 정규화가 끝난 5325개의 데이터들은 Batch Size=4, Epoch=100으로 YOLOv7-tiny를 통해 학습시켜

17가지의 행동 분류 결과를 텍스트로 추출한다.

![image](https://github.com/UiJoon64/seniorMotionDetection/assets/144432006/029870ad-8f2c-4897-99e3-c0d5e8e26a85)

</div>
</details>




🌟 영상데이터
====================
<details>
<summary>이전 영상 데이터</summary>
<div markdown="1">

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

</div>
</details>

위와 같은 방식은 피사체가 크게 변화할 때 추정된 골격의 오차가 매우 커져 대체 가능한 다른 방법을 찾아야 했다.

먼저, AI-hub에서 구한 영상 데이터는 2인 이상이 촬영되거나 피사체가 매우 작고 하이라이트가 길이에 비해 매우 짧은 편이었기에 수가 적고 전처리하기에 불편했다.

따라서 클래스 별로 영상을 직접 촬영하였다.

낙상에 대한 구분을 최우선적인 목표로 두었기에 낙상, 낙상과 가장 비슷한 행동 유형인 눕기, 배회와 같은 일상적인 행동 총 3가지로 구분하였다.

각 클래스에 대한 영상을 40개 정도로 확보하였고 각 영상은 약 3~4초 길이, 피사체 전체가 보이는 각도로 행동이 이루어지는 여러 경우의 수를 재현하여 촬영하였다.

촬영한 영상은 yolov7-w6-pose.pt 모델을 활용하여 골격의 위치를 추정, 추정한 2차원 좌표를 바탕으로 각도를 계산하였다.
(https://github.com/WongKinYiu/yolov7)

(가끔씩 유리창에 비친 모습도 골격을 추정해내는데, 한 프레임에 대한 모델의 출력값이 2배가 된다. 가장 신뢰도가 높은 값 골격을 추출해내면 여러 피사체가 있어도 목표 피사체에 대한 값만 가져올 수 있었다.)

pose-estimate.py는 디렉토리에 포함된 전체 영상에 대해 해당 과정을 수행하고 프레임 단위로 각 관절각도를 기록하여 csv파일로 생성한다.

골격 추정의 정확도를 확인하기 위해 골격이 그려진 영상을 같은 디렉토리에 저장되도록 하였지만 추출 과정에 상당한 시간이 추가 되기에 변환해야할 영상이 추가되어야 한다면 이 과정은 생략되더야 한다. (약 120개 영상 변환에 5시간 정도 소요)

이제 시간별 관절각도 변화 데이터를 전처리 해준다.

![Untitled](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/2d15e402-b0a5-4a06-a204-198e6c37d386)

하지만 촬영된 영상은 눕는 행위나 낙상 전후의 대기 동작까지 포함하기에 모두 같은 라벨링을 하면 정확도에 영향을 미치게 된다.

이를 방지하기 위해 구간별로 표준편차를 계산하여 4가지 관절 모두 변화가 크게 없는 구간은 타겟이 아니라고 판정, 제외하여 하이라이트만 추출한다.

![Untitled (1)](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/bae67576-7ab4-4634-8c83-2bfdcd708423)

위와 같이 구간별로 표준 편차를 추출한 다음,

표준 편차가 임계값 이하인 부분에 대해 필터링하면 다음과 같이 행의 개수가 줄어든, 하이라이트만 추출해낼 수 있다.

![download (2)](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/adec54c5-b7ff-4cbd-903c-87284acf3428)

![download (3)](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/d33b67df-d209-485a-b06d-833fc296ebad)

위와 같이 하이라이트만 추출된 것을 확인이 가능하다.

또, 최종 모니터링 프로그램은 1초에 10회씩 동작을 추정할 것이므로 30fps로 촬영된 훈련 데이터도 average pooling을 사용하여 10fps로 맞춰준다.

![Untitled (2)](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/bb7a9616-279a-4fff-b95a-22d18dc47228)

이렇게 전처리가 끝난 데이터 사용하여 원핫 인코딩을 사용한 타겟 데이터와 함께 훈련 데이터셋으로 만든다.

길이가 5인 시계열 데이터이기에 lstm층을 사용한 모델을 사용하는데, 정확도 향상을 위해서 dropout, batch normalization층을 활용해보았고

실험적으로나 이론적으로나 batch normalization층 만을 사용하는 것이 성능이 좋아서 BN층만 사용하였다(Batch Normalized Recurrent Neural Networks,2015).

위 과정은 모두 preprocessing_sequential_datas.ipynb에서 처리하였다.

yolov7+mediapipe 에서 yolov7 pose estimation 모델로 변환해서 관절각도변화 추이로 정확도를 비교해보니 상당 부분 개선이 이루어진 것을 확인할 수 있었다.

![Untitled (3)](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/40766e78-aa65-4588-9991-8d6693f6ebbd)

[좌 : yolov7+mediapipe, 우 : yolov7-pose-estimation]

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

로그인 페이지의 경우 firebase 서버와 연동하여 회원 정보를 관리하였다. 


🤩 프로젝트 결과 및 실행
========================

실행 파일을 생성할 때 docker를 사용하지않고 pyinstaller 라이브러리를 사용하였다.

'''
pyinstaller --onefile --add-data="templates;templates" --add-data="static;static" --add-data="models;models" --add-data="utils;utils" --add-data="latest_sequential_model.h5;." --add-data="yolov7-w6-pose.pt;." --hidden-import=seaborn --hidden-import=scipy.signal --hidden-import=matplotlib --hidden-import=matplotlib.pyplot --hidden-import=matplotlib.backends.backend_agg --hidden-import=yaml webcam_with_another_checking_skeleton_makingexe.py
'''

![Untitled (4)](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/b09cc619-3b40-43cd-a13e-2e9ce4ec31d6)

![Untitled (5)](https://github.com/UiJoon64/seniorMotionDetection/assets/117344692/cffdceff-91e8-4302-8d25-4c81537f88da)

