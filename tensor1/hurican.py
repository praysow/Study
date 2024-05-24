# # ======================================================================
# # There are 5 questions in this exam with increasing difficulty from 1-5.
# # Please note that the weight of the grade for the question is relative to its
# # difficulty. So your Category 1 question will score significantly less than
# # your Category 5 question.
# #
# # WARNING: Do not use lambda layers in your model, they are not supported
# # on the grading infrastructure. You do not need them to solve the question.
# #
# # You must use the Submit and Test button to submit your model
# # at least once in this category before you finally submit your exam,
# # otherwise you will score zero for this category.
# # ======================================================================
# #
# # COMPUTER VISION WITH CNNs
# #
# # Create and train a classifier to classify images between two classes
# # (damage and no_damage) using the satellite-images-of-hurricane-damage dataset.
# # ======================================================================
# #
# # ABOUT THE DATASET
# #
# # Original Source:
# # https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized
# # The dataset consists of satellite images from Texas after Hurricane Harvey
# # divided into two groups (damage and no_damage).
# # ==============================================================================
# #
# # INSTRUCTIONS
# #
# # We have already divided the data for training and validation.
# #
# # Complete the code in following functions:
# # 1. preprocess()
# # 2. solution_model()
# #
# # Your code will fail to be graded if the following criteria are not met:
# # 1. The input shape of your model must be (128,128,3), because the testing
# #    infrastructure expects inputs according to this specification. You must
# #    resize all the images in the dataset to this size while pre-processing
# #    the dataset.
# # 2. The last layer of your model must be a Dense layer with 1 neuron
# #    activated by sigmoid since this dataset has 2 classes.
# #
# # HINT: Your neural network must have a validation accuracy of approximately
# # 0.95 or above on the normalized validation dataset for top marks.
#
# import urllib
# import zipfile
#
# import tensorflow as tf
#
# # This function downloads and extracts the dataset to the directory that
# # contains this file.
# # DO NOT CHANGE THIS CODE
# # (unless you need to change https to http)
# def download_and_extract_data():
#     url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
#     urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
#     with zipfile.ZipFile('satellitehurricaneimages.zip', 'r') as zip_ref:
#         zip_ref.extractall()
#
# # This function normalizes the images.
# # COMPLETE THE CODE IN THIS FUNCTION
# def preprocess(image, label):
#     # NORMALIZE YOUR IMAGES HERE (HINT: Rescale by 1/.255)
#
#     return image, label
#
#
# # This function loads the data, normalizes and resizes the images, splits it into
# # train and validation sets, defines the model, compiles it and finally
# # trains the model. The trained model is returned from this function.
#
# # COMPLETE THE CODE IN THIS FUNCTION.
# def solution_model():
#     # Downloads and extracts the dataset to the directory that
#     # contains this file.
#     download_and_extract_data()
#
#     IMG_SIZE = 128
#     BATCH_SIZE = 64
#
#     # The following code reads the training and validation data from their
#     # respective directories, resizes them into the specified image size
#     # and splits them into batches. You must fill in the image_size
#     # argument for both training and validation data.
#     # HINT: Image size is a tuple
#     train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         directory='train/',
#         image_size=  # YOUR CODE HERE
#         , batch_size=BATCH_SIZE)
#
#     val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         directory='validation/',
#         image_size=  # YOUR CODE HERE
#         , batch_size=BATCH_SIZE)
#
#     # Normalizes train and validation datasets using the
#     # preprocess() function.
#     # Also makes other calls, as evident from the code, to prepare them for
#     # training.
#     # Do not batch or resize the images in the dataset here since it's already
#     # been done previously.
#     train_ds = train_ds.map(
#         preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
#         tf.data.experimental.AUTOTUNE)
#     val_ds = val_ds.map(
#         preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     # Code to define the model
#     model = tf.keras.models.Sequential([
#         # ADD LAYERS OF THE MODEL HERE
#
#         # If you don't adhere to the instructions in the following comments,
#         # tests will fail to grade your model:
#         # The input layer of your model must have an input shape of
#         # (128,128,3).
#         # Make sure your last layer has 1 neuron activated by sigmoid.
#         tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
#     ])
#
#     # Code to compile and train the model
#     model.compile(
#
#         # YOUR CODE HERE
#     )
#
#     model.fit(
#
#         # YOUR CODE HERE
#     )
#
#     return model
#
#
# # Note that you'll need to save your model as a .h5 like this.
# # When you press the Submit and Test button, your saved .h5 model will
# # be sent to the testing infrastructure for scoring
# # and the score will be returned to you.
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")

import urllib
import zipfile
import tensorflow as tf

# This function downloads and extracts the dataset to the directory that
# contains this file.
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
    urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
    with zipfile.ZipFile('satellitehurricaneimages.zip', 'r') as zip_ref:
        zip_ref.extractall()

# This function normalizes the images.
def preprocess(image, label):
    # Rescale by 1/.255 (Normalize the images)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()

    IMG_SIZE = 128
    BATCH_SIZE = 64

    # The following code reads the training and validation data from their
    # respective directories, resizes them into the specified image size
    # and splits them into batches.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    # Normalizes train and validation datasets using the preprocess() function.
    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20
    )

    return model

# Save the model as a .h5 file
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
