from ultralytics import YOLO
import cv2
import torch
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)
# YOLO 모델 로드
model = YOLO('best.pt')

model.model.names = ['빨간불','초록불','빨간불','초록불','자전거','킥보드','라바콘','횡단보도']
# 비디오 파일 로드
cap = cv2.VideoCapture('green.mp4')

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.5,vid_stride=7)

    confidence_thresholds = {
        # 'person': 0.7,
        # 'bicycle': 0.5,
        # 'car': 0.7,
        # 기타 클래스에 대한 임곗값 설정
    }
    # 탐지된 객체들을 순회하면서
    for det in results:
        box = det.boxes
        conf = box.conf
        for i in range(len(box.xyxy)):
            x1, y1, x2, y2 = box.xyxy[i].tolist()
            cv2.rectangle(det.orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # label = f"{box.cls[i]}:{box.conf[i]:.2f}"
            label = f"{box.cls[i]}"
            cv2.putText(det.orig_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x_center1 = (x1 + x2) / 2
            y_center1 = (y1 + y2) / 2
            if x_center1 < 720 / 2:
                x_center2 = 'left'
            elif 720 / 2 < x_center1 < 720 / 2 + 720 / 2:
                x_center2 = 'middle'
            elif 720 / 2 < x_center1:
                x_center2 = 'right'
            if y_center1 < 720 / 2:
                y_center2 = 'up'
            elif 720 / 2 < y_center1 < 720 / 2 + 720 / 2:
                y_center2 = 'center'
            elif 720 / 2 < y_center1:
                y_center2 = 'down'
            location = f"{x_center2} {y_center2}"
            print(f"{location}, {label}")

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


