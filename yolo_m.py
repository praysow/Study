import os
import cv2
from ultralytics import YOLO

# OpenMP 중복 로드 문제 해결
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# YOLO 모델 로드
model = YOLO("")

# 클래스별 이름 설정 (YOLO 모델에서 사용된 순서와 동일해야 함)
CLASS_NAMES = ['0', '1', '2', '3']

# 클래스 탐지 기록 저장
class2_detected_frames = []  # 클래스 2가 탐지된 프레임 번호 저장
recorded_class2_detections = []  # 기록된 클래스 2 탐지 정보 저장

# 비디오 파일 경로 및 출력 폴더 설정
video_path = ""
save_folder = ""
os.makedirs(save_folder, exist_ok=True)  # 저장 폴더 생성

# 프레임 번호 초기화
frame_number = 0

# 비디오에서 객체 탐지 및 조건 확인
results = model.track(source=video_path, show=True, stream=True)

for result in results:  # YOLO의 추적 결과 반복
    frame_number += 1  # 프레임 번호 수동 증가
    detections = result.boxes.data.cpu().numpy()  # 탐지 결과를 NumPy 배열로 변환
    frame = result.orig_img  # 현재 프레임 이미지
    
    # 현재 프레임에서 탐지된 클래스 추출
    detected_classes = [CLASS_NAMES[int(box[-1])] for box in detections]  # box[-1]은 클래스 ID
    
    # 클래스 2 탐지 여부 확인
    if '2' in detected_classes:
        class2_detected_frames.append(frame_number)

    # 클래스 3이 탐지되면 클래스 2의 조건 체크
    if '3' in detected_classes:
        # 클래스 2 탐지가 있었고, 100 프레임 이내인지 확인
        for detected_frame in class2_detected_frames:
            if 0 <= frame_number - detected_frame <= 100:
                recorded_class2_detections.append((detected_frame, frame_number))
                class2_detected_frames.remove(detected_frame)  # 기록된 탐지는 제거
                print(f"Class 2 recorded: Frame {detected_frame} (followed by Class 3 at Frame {frame_number})")
                
                # 장면 저장 (Class 2가 기록되었을 때 해당 프레임을 이미지로 저장)
                save_path = os.path.join(save_folder, f"class2_{detected_frame}_class3_{frame_number}.jpg")
                cv2.imwrite(save_path, frame)  # 현재 프레임을 이미지로 저장
                print(f"Saved: {save_path}")
                break
