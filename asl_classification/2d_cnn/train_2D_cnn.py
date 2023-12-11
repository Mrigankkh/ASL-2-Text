import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import pickle
# Assuming you have a dataset with features (86-length landmark arrays) and corresponding labels (1 of 26 alphabets)
# Replace X, y with your actual data

# Assuming X is an array with shape (number_of_samples, 86)
# and y is an array with shape (number_of_samples,)
# Example:
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
data_dict = pickle.load(open('data_2Dcnn.pickle', 'rb'))


# Inspect data
data = data_dict['data']
labels = np.asarray(data_dict['labels'])


lengths = [len(item) for item in data]
desired_length = 42

padded_data = [
    seq + [[0, 0]] * (desired_length - len(seq)) if len(seq) < desired_length else seq[:desired_length]
    for seq in data
]
data = np.array(padded_data)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True,random_state=42, stratify=labels)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode the labels
y_train_encoded = to_categorical(y_train_encoded)
y_test_encoded = to_categorical(y_test_encoded)
print("Y train encoded: ",y_train_encoded)
# Reshape the data to 3D for Conv1D layer
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', input_shape=(42, 2, 1)))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(filters=128, kernel_size=(3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model
model.fit(X_train_reshaped, y_train_encoded, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test_reshaped, y_test_encoded)
# print(f'Test Accuracy: {accuracy * 100:.2f}%')
f = open('cnn_model_2d.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

