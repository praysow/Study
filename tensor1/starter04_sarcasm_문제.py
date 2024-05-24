# # ======================================================================
# # There are 5 questions in this exam with increasing difficulty from 1-5.
# # Please note that the weight of the grade for the question is relative
# # to its difficulty. So your Category 1 question will score significantly
# # less than your Category 5 question.
# #
# # Don't use lambda layers in your model.
# # You do not need them to solve the question.
# # Lambda layers are not supported by the grading infrastructure.
# #
# # You must use the Submit and Test button to submit your model
# # at least once in this category before you finally submit your exam,
# # otherwise you will score zero for this category.
# # ======================================================================
# #
# # NLP QUESTION
# #
# # Build and train a classifier for the sarcasm dataset.
# # The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# # It will be tested against a number of sentences that the network hasn't previously seen
# # and you will be scored on whether sarcasm was correctly detected in those sentences.

# import json
# import tensorflow as tf
# import numpy as np
# import urllib
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# def solution_model():
#     url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
#     urllib.request.urlretrieve(url, 'sarcasm.json')

#     # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
#     vocab_size = 1000
#     embedding_dim = 16
#     max_length = 120
#     trunc_type='post'
#     padding_type='post'
#     oov_tok = "<OOV>"
#     training_size = 20000

#     sentences = []
#     labels = []
#     # YOUR CODE HERE


#     model = tf.keras.Sequential([
#     # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#     return model


# # Note that you'll need to save your model as a .h5 like this.
# # When you press the Submit and Test button, your saved .h5 model will
# # be sent to the testing infrastructure for scoring
# # and the score will be returned to you.
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # 이 코드는 변경하지 마세요. 테스트가 작동하지 않을 수 있습니다.
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    # 데이터셋 로드 및 전처리
    with open("sarcasm.json", 'r') as f:
        data = json.load(f)
        for item in data:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

    # 토크나이저 및 패딩 처리
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # 데이터 분할
    training_sentences = padded[:training_size]
    training_labels = np.array(labels[:training_size])
    testing_sentences = padded[training_size:]
    testing_labels = np.array(labels[training_size:])

    # 모델 구축
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 훈련
    history = model.fit(training_sentences, training_labels, epochs=100, validation_data=(testing_sentences, testing_labels), verbose=1)

    # 훈련 과정에서의 정확도 확인
    train_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']
    print("훈련 정확도:", train_accuracy)
    print("검증 정확도:", test_accuracy)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("sarcasm.h5")
'''
훈련 정확도: [0.5687999725341797, 0.761900007724762, 0.8147000074386597, 0.8240500092506409, 0.8326500058174133, 0.8342999815940857, 0.8358500003814697, 0.8385499715805054, 0.8392000198364258, 0.8397499918937683]
검증 정확도: [0.6299001574516296, 0.785362958908081, 0.8123416304588318, 0.8023550510406494, 0.819049060344696, 0.8191980719566345, 0.81293785572052, 0.8191980719566345, 0.8186019062995911, 0.8187509179115295]
'''