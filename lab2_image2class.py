import tensorflow as tf
import numpy as np
import cv2
import PIL.Image
import os


WITH_EVALUATED_LANDMARKS_BASE_PATH = 'data/with-evaluated-landmarks'


def pairwise(np_array):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    return np_array.reshape(len(np_array[0]) // 2, 2)


def load_data(file_name='data/images.npy'):
    image_data = np.load(file_name, allow_pickle=True)
    np.random.shuffle(image_data)
    return image_data


def split_data(image_data):
    images = []
    labels = []
    for (label, image, points) in image_data:
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


def split_train_test(images, labels):
    images_20percent_size = len(images) // 5
    labels_20percent_size = len(labels) // 5
    test_images = images[:images_20percent_size]
    test_labels = labels[:labels_20percent_size]
    train_images = images[images_20percent_size:]
    train_labels = labels[labels_20percent_size:]
    return train_images, train_labels, test_images, test_labels


def create_model():
    md = tf.keras.models.Sequential([
        # Copied from lab1
        tf.keras.layers.InputLayer(input_shape=[48, 48, 1]),
        tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2]),
        tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2]),
        tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2]),
        tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2]),
        tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', use_bias=False),
        tf.keras.layers.LeakyReLU(alpha=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(units=136),
        # Some addition to get a classifier
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    md.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
    return md


np_image_data = load_data()
np_images, np_labels = split_data(np_image_data)
np_train_images, np_train_labels, np_test_images, np_test_labels = split_train_test(np_images, np_labels)

model = create_model()

model.fit(np_train_images, np_train_labels, validation_split=.1, batch_size=64, epochs=100)

print("Evaluating on test data:")
model.evaluate(np_test_images, np_test_labels)
model.save('models/lab2/lab2_image2class_1.keras')
