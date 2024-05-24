# import tensorflow as tf
# print(tf.__version__)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if(gpus):
#     print('gpu move!!')
# else:
#     print('gpu don`t move!!!')
#

import torch

# GPU가 사용 가능한지 확인
if torch.cuda.is_available():
    # 현재 사용 중인 GPU 디바이스 확인
    device = torch.device("cuda")
    print("GPU가 사용됩니다.")
    print("사용 중인 GPU:", torch.cuda.get_device_name(0))  # 여러 개의 GPU가 있는 경우 첫 번째 GPU의 이름을 출력합니다.
else:
    device = torch.device("cpu")
    print("GPU를 사용할 수 없습니다. CPU가 사용됩니다.")

# Tensor를 GPU로 이동
tensor = torch.tensor([1, 2, 3]).to(device)
