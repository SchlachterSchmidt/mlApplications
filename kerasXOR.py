from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[0, 0, 1],
				[0, 1, 1],
				[1, 0, 1],
				[0, 1, 0],
				[1, 0, 0],
				[1, 1, 1],
				[0, 0, 0]])

target  = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

print(data)
print(target)

model = Sequential()

model.add(Dense(300, activation='relu', input_shape=(3,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
				metrics=['accuracy'])

history = model.fit(data, target, epochs=75)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy and loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

test_array = np.array([[1, 1, 0]])
predictions = model.predict(test_array)
print('-'*30)
print('Predicted output for [1,1,0]')
print('{:.10f}'.format(predictions[0][0]))

print('-'*30)
print('Predictions for supplied inputs using our new model')
print(model.predict_proba(data))
