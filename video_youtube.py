

from ultralytics import YOLO
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyttsx3
from queue import Queue
from threading import Thread
import yt_dlp
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# YOLO 모델 로드
# model = YOLO(r'C:\project\yolov8\test.pt')
model = YOLO(r'C:\project\yolov8\runs\detect\yolo8m_img640_batch16\weights\best.pt')
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 클래스별 바운딩 박스 색상 지정
colors = {
    '빨간불': (255, 0, 0),  # 빨간색
    '초록불': (0, 255, 0),  # 초록색
    '자전거': (0, 0, 0),  # 검정색
    '킥보드': (128, 0, 128),  # 보라색
    '라바콘': (255, 165, 0),  # 주황색
    '횡단보도': (255, 255, 255)  # 흰색
}

# 한글 폰트 경로 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 20)

# TTS 엔진 초기화 (한 번만 실행)
engine = pyttsx3.init()
engine.setProperty('rate', 250)  # TTS 속도 조절

# TTS 큐 및 스레드 설정
tts_queue = Queue()
previous_objects = set()
last_tts_time = 0  # 마지막 TTS 호출 시간
TTS_INTERVAL = 10  # TTS 호출 간격 (초)

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        speak(text)
        tts_queue.task_done()

def speak(text):
    engine.say(text)
    engine.runAndWait()

tts_thread = Thread(target=tts_worker)
tts_thread.start()

# 유튜브 비디오 스트림 로드
url = 'https://www.youtube.com/shorts/g-McYJxv08I'

ydl_opts = {
    'format': 'best',
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_url = info_dict['url']

cap = cv2.VideoCapture(video_url)

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model(frame)
    boxes = results[0].boxes

    # 프레임을 PIL 이미지로 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    detected_objects = set()

    # 탐지된 객체들을 순회하면서
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.tolist()[0]
        cls_id = int(box.cls)
        conf = box.conf.item()

        # 클래스 ID가 model.model.names의 범위를 벗어나지 않도록 확인
        if cls_id < len(model.model.names):
            label = model.model.names[cls_id]
            color = colors.get(label, (128, 128, 128))  # 기본 색상은 회색
        else:
            label = "Unknown"
            color = (128, 128, 128)  # 기본 색상은 회색

        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), f"{label}:{conf:.2f}", font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, y1 - text_height - 10), f"{label}:{conf:.2f}", font=font, fill=color)

        # 중심 좌표 계산 및 위치 표시
        x_center1 = (x1 + x2) / 2
        y_center1 = (y1 + y2) / 2
        if x_center1 < 720 / 2:
            x_center2 = '좌'
        elif 720 / 2 < x_center1 < 720:
            x_center2 = '정면'
        else:
            x_center2 = '우'
        if y_center1 < 720 / 2:
            y_center2 = '상'
        elif 720 / 2 < y_center1 < 720:
            y_center2 = '중'
        else:
            y_center2 = '하'
        location = f"{x_center2} {y_center2}"
        detected_objects.add(f"{location}, {label}")

    # 상태 변경 감지 및 TTS 큐에 텍스트 추가
    current_time = time.time()
    if detected_objects != previous_objects and current_time - last_tts_time > TTS_INTERVAL:
        unique_changes = detected_objects - previous_objects
        for change in unique_changes:
            tts_queue.put(change)
            print(change)
        previous_objects = detected_objects
        last_tts_time = current_time

    # PIL 이미지를 다시 OpenCV 이미지로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# TTS 스레드 종료
tts_queue.put(None)
tts_thread.join()

cap.release()
cv2.destroyAllWindows()
