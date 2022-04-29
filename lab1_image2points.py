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
    landmarks = []
    for (imageType, image, points) in image_data:
        images.append(image)
        landmarks.append(points.flatten())
    return np.array(images), np.array(landmarks)


def split_train_test(images, landmarks):
    images_20percent_size = len(images) // 5
    landmarks_20percent_size = len(landmarks) // 5
    test_images = images[:images_20percent_size]
    test_landmarks = landmarks[:landmarks_20percent_size]
    train_images = images[images_20percent_size:]
    train_landmarks = landmarks[landmarks_20percent_size:]
    return train_images, train_landmarks, test_images, test_landmarks


def create_model(output_n=136):
    model = tf.keras.models.Sequential([
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
        tf.keras.layers.Dense(units=output_n),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error",
                  metrics=['mae', 'accuracy'])
    return model


np_image_data = load_data()
np_images, np_landmarks = split_data(np_image_data)
np_train_images, np_train_landmarks, np_test_images, np_test_landmarks = split_train_test(np_images, np_landmarks)

model_136 = create_model(136)
# model_136 = tf.keras.models.load_model('models/lab1/lab1_3.keras')

model_136.fit(np_train_images, np_train_landmarks, validation_split=.1, batch_size=64, epochs=1000)

print("Evaluating on test data:")
model_136.evaluate(np_test_images, np_test_landmarks)
model_136.save('models/lab1/lab1_4.keras')

print("Evaluating landmarks on images...")

image_counter = 0
for (image_type, image, points) in np_image_data:
    image_expanded = np.expand_dims(image, axis=0)
    predicted_shape = model_136.predict(image_expanded)
    predicted_shape_int = predicted_shape.astype(int)
    pairwise_predicted_shape = pairwise(predicted_shape_int)
    for x, y in pairwise_predicted_shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    image_with_landmarks = PIL.Image.fromarray(image)
    image_dir = os.path.join(WITH_EVALUATED_LANDMARKS_BASE_PATH, str(image_type))
    os.makedirs(image_dir, exist_ok=True)
    image_with_landmarks.save(os.path.join(image_dir, str(image_counter) + '.png'))
    image_counter += 1
