import tensorflow as tf

tf.compat.v1.enable_eager_execution() #즉시실행모드 1.0

print('tensorflow_version',tf.__version__)
print('즉시실행모드',tf.executing_eagerly())
# tensorflow_version 1.14.0
# 즉시실행모드 True

tf.compat.v1.disable_eager_execution()  #즉시실행모드 2.0

print('tensorflow_version',tf.__version__)
print('즉시실행모드',tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
#tensorflow 1버전
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)
else :
    print('gpu 없음')

#tensorflow 2버전
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         print(gpus[0])
#     except RuntimeError as e:
#         print(e)
# else:
#     print('gpu 없음')
    
