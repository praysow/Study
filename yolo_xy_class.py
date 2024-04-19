import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 루트 디렉토리
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # PATH에 ROOT 추가
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 상대 경로

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from gtts import gTTS
from playsound import playsound
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)

from utils.torch_utils import select_device, smart_inference_mode
def text_to_speech(text):
    if text:  # 텍스트가 비어있지 않을 경우에만 TTS 모듈 호출
        tts = gTTS(text=text, lang='en')
        output_path = f'sound/xy_class2.mp3'
        tts.save(output_path)
        return output_path
    else:
        return None  # 텍스트가 비어있을 경우 None 반환


@smart_inference_mode()
def run(
    weights=ROOT / "yolov8n.pt",  # 모델 경로 또는 triton URL
    source=ROOT / "data/images",  # 파일/디렉토리/URL/글로브/스크린/0(웹캠)
    data=ROOT / "data/coco128.yaml",  # 데이터셋.yaml 경로
    imgsz=(640, 640),  # 추론 크기 (높이, 너비)
    conf_thres=0.25,  # 신뢰 임계값
    iou_thres=0.45,  # NMS IOU 임계값
    max_det=1000,  # 이미지 당 최대 감지 수
    device="",  # cuda 장치, 예: 0 또는 0,1,2,3 또는 cpu
    view_img=False,  # 결과 표시
    save_txt=False,  # 결과를 *.txt로 저장
    save_csv=False,  # CSV 형식으로 결과 저장
    save_conf=False,  # --save-txt 레이블에 확률 저장
    save_crop=False,  # 잘린 예측 상자 저장
    nosave=False,  # 이미지/비디오 저장 안 함
    classes=None,  # 클래스별로 필터링: --class 0 또는 --class 0 2 3
    agnostic_nms=False,  # 클래스에 무관한 NMS
    augment=False,  # 증강된 추론
    visualize=False,  # 특징 시각화
    update=False,  # 모든 모델 업데이트
    project=ROOT / "runs/detect",  # 결과를 project/name에 저장
    name="exp",  # 결과를 project/name에 저장
    exist_ok=False,  # 기존 project/name 허용, 증가 안 함
    line_thickness=3,  # 바운딩 박스 두께 (픽셀)
    hide_labels=False,  # 라벨 숨기기
    hide_conf=False,  # 신뢰도 숨기기
    half=False,  # FP16 하프-프리시전 추론 사용
    dnn=False,  # ONNX 추론에 OpenCV DNN 사용
    vid_stride=1,  # 비디오 프레임 속도 간격
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # 추론 이미지 저장
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # 다운로드

    # 디렉토리
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 실행 증가
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 디렉토리 만들기

    # 모델 로드
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 이미지 크기 확인

    # 데이터로더
    bs = 1  # 배치 크기
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 추론 실행
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 워밍업
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8에서 fp16/32로
            im /= 255  # 0 - 255에서 0.0 - 1.0으로 변환
            if len(im.shape) == 3:
                im = im[None]  # 배치 차원 확장
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # 추론
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 두 번째 스테이지 분류기 (선택 사항)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # CSV 파일 경로 정의
        csv_path = save_dir / "predictions.csv"

        # CSV 파일에 작성 또는 추가
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # 예측 처리
        for i, det in enumerate(pred):  # 이미지 당
            seen += 1
            if webcam:  # 배치 크기 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                # s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # 경로로 변환
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            # s += "%gx%g " % im.shape[2:]  # 문자열 추가
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 정규화 이득 whwh
            imc = im0.copy() if save_crop else im0  # 잘린 예측 상자에 대해
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 상자 크기를 img_size에서 im0 크기로 변경
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 결과 출력
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 클래스별 감지 수
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 문자열 추가

                # 결과 쓰기
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 정수 클래스
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # 파일에 작성
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 정규화 xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 레이블 형식
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # 이미지에 상자 추가
                        c = int(cls)  # 정수 클래스
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
#######################################################################x,y 위치값############################################################################
                    # x_center와 y_center를 먼저 None으로 초기화
                    x_center = None
                    y_center = None

                    # 이미지의 각 예측에 대해 반복
                    for *xyxy, conf, cls in reversed(det):
                        if not xyxy:  # 위치값이 없으면 건너뜁니다
                            continue
                        
                        c = int(cls)  # 정수 클래스
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # 파일에 작성
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 정규화 xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 레이블 형식
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # 이미지에 상자 추가
                            c = int(cls)  # 정수 클래스
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                        # 객체의 중심점 계산
                        x_center = (xyxy[0] + xyxy[2]) / 2
                        y_center = (xyxy[1] + xyxy[3]) / 2
                        if 0 <= x_center <= 212:
                            x_direction = "left"
                        elif 212 < x_center <= 424:
                            x_direction = "front"
                        else:
                            x_direction = "right"

                        if 0 <= y_center <= 212:
                            y_direction = "down"
                        elif 212 < y_center <= 424:
                            y_direction = "center"
                        else:
                            y_direction = "up"
###########################################################################로그 출력되는 부분####################################################################
                        # x_center와 y_center가 모두 None이 아닌 경우에만 실행합니다
                        # if x_center is not None and y_center is not None:
                        #     if text_to_speech(f"X: {x_center}, Y: {y_center},{names[int(c)]}"):
                        #         playsound(text_to_speech(f"X: {x_center}, Y: {y_center},{names[int(c)]}"))
                        #         print(f"X: {x_center}, Y: {y_center},{n} {names[int(c)]}{'s' * (n > 1)}")
###########################################################################로그를 TTS로 변환####################################################################
                        if x_center is not None and y_center is not None:
                            if text_to_speech(f"{x_direction}{y_direction},{names[int(c)]}"):
                                playsound(text_to_speech(f"{x_direction}{y_direction},{names[int(c)]}"))
                                print(f"{x_direction}{y_direction},{n} {names[int(c)]}{'s' * (n > 1)}")
            
            # 결과 스트림
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 창 크기 조정 허용 (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 밀리초

            # 결과 저장 (감지된 이미지)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' 또는 'stream'
                    if vid_path[i] != save_path:  # 새 비디오
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 이전 비디오 라이터 해제
                        if vid_cap:  # 비디오
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 스트림
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # 결과 비디오에 *.mp4 접미사 강제 적용
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
            

    # 결과 출력
    t = tuple(x.t / seen * 1e3 for x in dt)  # 이미지당 속도
    # LOGGER.info(f"속도: %.1fms 전처리, %.1fms 추론, %.1fms NMS 이미지당 {(1, 3, *imgsz)}의 모양에서" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 레이블이 {save_dir / 'labels'}에 저장됨" if save_txt else ""
        LOGGER.info(f"결과가 {colorstr('bold', save_dir)}에 저장됨{s}")
    if update:
        strip_optimizer(weights[0])  # 모델 업데이트 (SourceChangeWarning 수정)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="모델 경로 또는 triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "video/demo.mp4", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml 경로")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="추론 크기 h,w")
    parser.add_argument("--conf-thres", type=float, default=0.9, help="신뢰 임계값")    #신뢰도가 몇 이상인 객체만 인식할지 결정
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 임계값") #몇 이하의 신뢰도의 객체가 겹치면 하나로 인식
    parser.add_argument("--max-det", type=int, default=1000, help="이미지 당 최대 감지 수")
    parser.add_argument("--device", default="", help="cuda 장치, 예: 0 또는 0,1,2,3 또는 cpu")
    parser.add_argument("--view-img", action="store_true", help="결과 표시")
    parser.add_argument("--save-txt", action="store_true", help="결과를 *.txt로 저장")
    parser.add_argument("--save-csv", action="store_true", help="결과를 CSV 형식으로 저장")
    parser.add_argument("--save-conf", action="store_true", help="--save-txt 레이블에 확률 저장")
    parser.add_argument("--save-crop", action="store_true", help="잘린 예측 상자 저장")
    parser.add_argument("--nosave", action="store_true", help="이미지/비디오 저장 안 함")
    parser.add_argument("--classes", nargs="+", type=int, help="클래스별로 필터링: --classes 0 또는 --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="클래스에 무관한 NMS")
    parser.add_argument("--augment", action="store_true", help="증강된 추론")
    parser.add_argument("--visualize", action="store_true", help="특징 시각화")
    parser.add_argument("--update", action="store_true", help="모든 모델 업데이트")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="결과를 project/name에 저장")
    parser.add_argument("--name", default="exp", help="결과를 project/name에 저장")
    parser.add_argument("--exist-ok", action="store_true", help="기존 project/name 허용, 증가 안 함")
    parser.add_argument("--line-thickness", default=3, type=int, help="바운딩 박스 두께 (픽셀)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="라벨 숨기기")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="신뢰도 숨기기")
    parser.add_argument("--half", action="store_true", help="FP16 하프-프리시전 추론 사용")
    parser.add_argument("--dnn", action="store_true", help="ONNX 추론에 OpenCV DNN 사용")
    parser.add_argument("--vid-stride", type=int, default=1, help="비디오 프레임 속도 간격")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 확장
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


