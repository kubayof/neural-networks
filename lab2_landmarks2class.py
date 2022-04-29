import numpy as np
import tensorflow as tf


def load_data(file_name='data/images.npy'):
    np_image_data = np.load(file_name, allow_pickle=True)
    np.random.shuffle(np_image_data)
    landmarks = []
    labels = []
    for (label, image, points) in np_image_data:
        landmarks.append(points.reshape(136, 1))
        labels.append(label)
    return np.array(landmarks), np.array(labels)


def split_train_test(landmarks, labels):
    landmarks_20percent_size = len(landmarks) // 5
    labels_20percent_size = len(labels) // 5
    test_landmarks = landmarks[:landmarks_20percent_size]
    test_labels = labels[:labels_20percent_size]
    train_landmarks = landmarks[landmarks_20percent_size:]
    train_labels = labels[labels_20percent_size:]
    return train_landmarks, train_labels, test_landmarks, test_labels


def create_model():
    md = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[136, 1]),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    md.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
    return md


np_landmarks, np_labels = load_data()
np_train_landmarks, np_train_labels, np_test_landmarks, np_test_labels = split_train_test(np_landmarks, np_labels)
model = create_model()
model.fit(np_train_landmarks, np_train_labels, batch_size=64, epochs=1000)

print("Evaluating on test data:")
model.evaluate(np_test_landmarks, np_test_labels)

model.save('models/lab2/lab2_1.keras')
