import tensorflow as tf
import numpy as np

def generate_data(num_samples=1000, num_bits=32):  
    X = np.random.randint(0, 2, size=(num_samples, 32, 2))  
    Y = np.sum(X, axis=1)
    return X, Y

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(16, input_shape=(32, 2), activation='relu', return_sequences=True),  
    tf.keras.layers.SimpleRNN(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

X_train, Y_train = generate_data()

model.fit(X_train, Y_train, epochs=10, batch_size=32)

X_test, Y_test = generate_data(10)
predictions = model.predict(X_test)

for i in range(10):
    input_data = X_test[i]
    true_output = Y_test[i]
    predicted_output = predictions[i].round()

    print(f"Wejscie: {input_data}")
    print(f"Prawdziwa suma: {true_output}")
    print(f"Przewidziana suma: {predicted_output}")
    print()
