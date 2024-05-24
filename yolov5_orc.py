'''
from google.colab import drivedrive.mount('/content/gdrive')

%cd /content/gdrive/MyDrive/yolov5!pip install -r requirements.txt

pip install wandb

import os
os.mkdir('/content/busbusbus')

%cd /content/busbusbus
!unzip -qq '/content/gdrive/MyDrive/data/BUS_Final_Index_Cor.zip'

%cd /content
!unzip -qq '/content/gdrive/MyDrive/data/MergedDataset.zip'
'''

import shutil
import numpy as np
from glob import glob
import os

cross_imgs_tr = os.listdir('/content/train/images')
cross_imgs_tr = np.random.choice(cross_imgs_tr, 7500, False)

label_total = os.listdir('/content/train/labels/')

cross_labels_tr = []

for i in label_total :
    if i[:-4]+'.jpg' in cross_imgs_tr :
        cross_labels_tr.append(i)

    elif i[:-4]+'.png' in cross_imgs_tr :
        cross_labels_tr.append(i)
        
for img, label in zip(cross_imgs_tr, cross_labels_tr) :

    if img in os.listdir('/content/busbusbus/train/images') :
        continue

    elif label in os.listdir('/content/busbusbus/train/labels') :
        continue


    shutil.move(f'/content/train/images/{img}', '/content/busbusbus/train/images')
    shutil.move(f'/content/train/labels/{label}', '/content/busbusbus/train/labels')
    
cross_imgs_val = os.listdir('/content/valid/images')
cross_imgs_val = np.random.choice(cross_imgs_val, 600, False)


label_total = os.listdir('/content/valid/labels/')

cross_labels_val = []

for i in label_total :
    if i[:-4]+'.jpg' in cross_imgs_val :
        cross_labels_val.append(i)

    elif i[:-4]+'.png' in cross_imgs_val :
        cross_labels_val.append(i)
        
for img, label in zip(cross_imgs_val, cross_labels_val) :

    if img in os.listdir('/content/busbusbus/valid/images') :
        continue

    elif label in os.listdir('/content/busbusbus/valid/labels') :
        continue


    shutil.move(f'/content/valid/images/{img}', '/content/busbusbus/valid/images')
    shutil.move(f'/content/valid/labels/{label}', '/content/busbusbus/valid/labels')
    

# print('Train 이미지 수 : ',len(list(glob('/content/busbusbus/train/images/*'))))
# print('Train 라벨 수 : ',len(list(glob('/content/busbusbus/train/labels/*'))))


# print('Valid 이미지 수 : ',len(list(glob('/content/busbusbus/valid/images/*'))))
# print('Valid 라벨 수 : ',len(list(glob('/content/busbusbus/valid/labels/*'))))

# Train 이미지 수 :  14838
# Train 라벨 수 :  14838
# Valid 이미지 수 :  1211
# Valid 라벨 수 :  1211

# python /content/gdrive/MyDrive/yolov5/train.py --img 640 --batch 16 --epochs 20 --data /content/FinalYaml.yaml --cfg /content/gdrive/MyDrive/yolov5/models/yolov5l.yaml --weights yolov5l.pt --name _FINAL_YOLO_20

# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.3 --source /content/gdrive/MyDrive/20150208124810_hysghyhc.jpg --hide-conf

# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/DSC_7307.jpg 

# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/Yolo_for_Infer/IMG_7064.MOV --hide-conf

# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/Yolo_for_Infer/IMG_7065.JPG --hide-conf
     
# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/Yolo_for_Infer/IMG_7066.MOV --hide-conf
     
# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/Yolo_for_Infer/IMG_7067.MOV --hide-conf

# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/Yolo_for_Infer/IMG_7068.MOV --hide-conf
     
# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/Yolo_for_Infer/Youtube_BUS.mp4 --hide-conf
     
# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/Yolo_for_Infer/IMG_7085.MOV --hide-conf

# python /content/gdrive/MyDrive/yolov5/detect.py --weights /content/gdrive/MyDrive/yolov5/runs/train/_FINAL_YOLO_20/weights/best.pt --img 640 --conf 0.25 --source /content/gdrive/MyDrive/DSC_7307.jpg --hide-conf
     