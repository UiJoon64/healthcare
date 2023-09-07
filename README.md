# 시니어 이상행동 감지 서비스

### 정상 실행부터 시작

webcam_with_another.py 파일과 Yolov7-Flask 폴더 속 app.py 파일 정상 실행부터 하는게 목적

webcam_with_another.py는 웹캠 카메라 분석 페이지가 구현되어있지만 sql연결, 로그인 페이지 구현 X

app.py파일은 결과 페이지가 구현 X

따로 개발 후 나중에 합치는 것으로 예정


파일 실행 시 필요한 경로, 빈 폴더가 부족할 수 있음

확인 필요

### requirements

CUDA 11.3(그래픽 카드 드라이버 버전에 따라 상이) 설치 후 쿠다 버전에 맞는 파이토치 설치 \
https://pytorch.org/get-started/previous-versions/ 에서 확인
```
# CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

이후 tensorflow까지 설치

### 설치 과정에서 Error
```
ImportError: cannot import name 'builder' from 'google.protobuf.internal'
```

https://velog.io/@alvinlee9/ImportError-cannot-import-name-builder-from-google.protobuf.internal 참고



# yolov7-object-cropping

### Steps to run Code

```
- Clone the repository.
```
git clone https://github.com/noorkhokhar99/yolov7-object-cropping.git
```
- Goto the cloned folder.
```
cd yolov7-object-cropping
```

```
- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Download [yolov7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) object detection weights from link and move them to the working directory {yolov7-object-cropping}
- Run the code with mentioned command below.
```
#if you want to change source file
python detect_and_crop.py --weights yolov7.pt --source "your video.mp4"

#for specific class (person)
python detect_and_crop.py --weights yolov7.pt --source "your video.mp4" -classes 0
```
- Cropped Objects will be stored in "working-dir/crop" folder.

### Results


<img src="https://github.com/noorkhokhar99/yolov7-object-cropping/blob/main/Screen%20Shot%201444-03-29%20at%201.34.23%20PM.png">

