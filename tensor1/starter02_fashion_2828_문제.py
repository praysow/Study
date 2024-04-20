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
# # Basic Datasets Question
# #
# # Create a classifier for the Fashion MNIST dataset
# # Note that the test will expect it to classify 10 classes and that the
# # input shape should be the native size of the Fashion MNIST dataset which is
# # 28x28 monochrome. Do not resize the data. Your input layer should accept
# # (28,28) as the input shape only. If you amend this, the tests will fail.
# #
# import tensorflow as tf


# def solution_model():
#     fashion_mnist = tf.keras.datasets.fashion_mnist

#     # YOUR CODE HERE
#     return model


# # Note that you'll need to save your model as a .h5 like this.
# # When you press the Submit and Test button, your saved .h5 model will
# # be sent to the testing infrastructure for scoring
# # and the score will be returned to you.
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")


import tensorflow as tf

def solution_model():
    # Load Fashion MNIST dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize pixel values to range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss:', loss)
    print('accuracy:', accuracy)
    return model

# Save the model
if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")
'''
loss: 0.34526005387306213
accuracy: 0.8758000135421753
'''