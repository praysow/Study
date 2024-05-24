# 객체 정보를 저장할 리스트 초기화
object_details = []

# 이미지의 각 예측에 대해 반복
for *xyxy, conf, cls in reversed(det):
    if not xyxy:  # 위치값이 없으면 건너뜁니다
        continue
    
    c = int(cls)  # 정수 클래스
    label = names[c]  # 클래스 이름
    confidence = float(conf)
    
    if save_csv:
        confidence_str = f"{confidence:.2f}"
        write_to_csv(p.name, label, confidence_str)

    if save_txt:  # 파일에 작성
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 정규화 xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(f"{txt_path}.txt", "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")

    if save_img or save_crop or view_img:  # 이미지에 상자 추가
        annotator.box_label(xyxy, label + f" {conf:.2f}", color=colors(c, True))
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

    # 중심점과 클래스 정보를 리스트에 추가
    object_details.append((x_direction, y_direction, label))

# 모든 객체의 중심점과 클래스 정보가 object_details 리스트에 저장됩니다.
print(object_details)
