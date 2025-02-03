import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

def generate_data(num_samples=1000, sequence_length=10, num_classes=2):
    X = np.random.rand(num_samples, sequence_length, 1)  
    y = np.random.randint(0, num_classes, num_samples)   
    return X, tf.keras.utils.to_categorical(y, num_classes)

def build_rnn_model(input_shape, num_classes):
    model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=input_shape, return_sequences=False),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

X_train, y_train = generate_data(num_samples=800, sequence_length=10)
X_test, y_test = generate_data(num_samples=200, sequence_length=10)

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y_train.shape[1]

model = build_rnn_model(input_shape, num_classes)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy:.4f}')

predictions = model.predict(X_test[:5])
print("Predictions (first 5 samples):", np.argmax(predictions, axis=1))
