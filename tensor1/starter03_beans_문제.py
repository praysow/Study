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
# # Computer Vision with CNNs
# # For this exercise you will use the beans dataset from TFDS
# # to build a classifier that recognizes different types of bean disease
# # Please make sure you keep the given layers as shown, or your submission
# # will fail to be graded. Please also note the image size of 224x224


# import tensorflow as tf
# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()




# def map_data(image, label, target_height = 224, target_width = 224):
#     """Normalizes images: `unit8` -> `float32` and resizes images
#     by keeping the aspect ratio the same without distortion."""
#     image = # Your Code here to normalize the image
#     image = tf.image.resize_with_crop_or_pad(# Parameters to resize and crop the image as desired)
#     return # Return the appropriate parameters

# def solution_model():
#     (ds_train, ds_validation, ds_test), ds_info = tfds.load(
#         name=#Dataset sames,
#         split=[#Desired Splits],
#         as_supervised=#Appropriate parameter,
#         with_info=#Appropriate parameter)


#     ds_train = # Perform appropriate operations to prepare ds_train

#     ds_validation = # Perform appropriate operations to prepare ds_validation

#     ds_test = # Perform appropriate operations to prepare ds_test

#     model = tf.keras.models.Sequential([
#       # You can change any parameters here *except* input_shape
#       tf.keras.layers.Conv2D(16, (3, 3), input_shape=(224, 224, 3), strides=2, padding='same', activation = 'relu'),
#       # Add whatever layers you like
#       # Keep this final layer UNCHANGED
#       tf.keras.layers.Dense(3, activation='softmax'),
#     ])

#     model.compile(
#         # Choose appropriate parameters
#     )

#     history = model.fit(
#         # Choose appropriate parameters
#     )
#     return model

# if __name__ == '__main__':
#     model = solution_model()
#     model.save("c3q4.h5")

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

def map_data(image, label, target_height=224, target_width=224):
    """Normalizes images: `uint8` -> `float32` and resizes images
    by keeping the aspect ratio the same without distortion."""
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
    image = tf.image.resize_with_crop_or_pad(image, target_height, target_width)  # Resize the image
    return image, label

def solution_model():
    (ds_train, ds_validation, ds_test), ds_info = tfds.load(
        name='beans',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        as_supervised=True,
        with_info=True
    )

    ds_train = ds_train.map(map_data)
    ds_train = ds_train.shuffle(buffer_size=1000).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_validation = ds_validation.map(map_data)
    ds_validation = ds_validation.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(map_data)
    ds_test = ds_test.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=(224, 224, 3), strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(ds_train, validation_data=ds_validation, epochs=10)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("c3q4.h5")


