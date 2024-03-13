import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())

# 즉시실행모드 -> 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행시킨다.
# 즉시실행모드 켠다
tf.compat.v1.disable_eager_execution()  #즉시실행모드 종료 //텐서플로 1.0문법// 디폴트
# tf.compat.v1.enable_eager_execution()   #즉시실행모드 실행 //텐서플로 2.0사용가능

print(tf.executing_eagerly())

hello = tf.constant('hello')

sess = tf.compat.v1.Session()       #sess 홀더는 입력만해줌
print(sess.run(hello))

#가상환경       즉시실행모드
#1.14.0         disable(디폴트)  사용가능   ★★★★
#1.14.0         enable           에러
#2.9.0          disable          사용가능   ★★★★
#2.9.0          enable(디폴트)   에러