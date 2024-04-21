# 객체의 중심점을 저장할 리스트 초기화
centers = []

# 이미지의 각 예측에 대해 반복
for *xyxy, conf, cls in reversed(det):
    if not xyxy:  # 위치값이 없으면 건너뜁니다
        continue
    
    c = int(cls)  # 정수 클래스
    label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
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
        annotator.box_label(xyxy, label, color=colors(c, True))
    if save_crop:
        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

    # 객체의 중심점 계산
    x_center = (xyxy[0] + xyxy[2]) / 2
    y_center = (xyxy[1] + xyxy[3]) / 2
    centers.append((x_center, y_center))  # 중심점을 리스트에 추가

# 모든 객체의 중심점 정보가 centers 리스트에 저장됩니다.
print(centers)
