import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
print(data_dict)

# Inspect data
data = data_dict['data']
lengths = [len(item) for item in data]

desired_length = 84
padded_data = [seq + [0] * (desired_length - len(seq)) if len(seq) < desired_length else seq[:desired_length] for seq in data]

data = np.array(padded_data)


# Convert to NumPy array (if data structure is uniform)
# data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Model training
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Prediction
y_predict = model.predict(x_test)

# Evaluate accuracy (optional)
accuracy = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly !'.format(accuracy*100))


f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()



# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras.optimizers import Adam
#
# # Define the root directory of your dataset
# dataset_root = './data2'  # Change this to your dataset directory
#
# # Set the image size and batch size
# sz = 128
# batch_size = 10
#
# # Create ImageDataGenerator instances for train and validation data
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2)  # Use validation_split to specify the split ratio
#
# # Load and preprocess data directly from the dataset root directory
# train_generator = train_datagen.flow_from_directory(
#     dataset_root,
#     target_size=(sz, sz),
#     batch_size=batch_size,
#     color_mode='grayscale',
#     class_mode='categorical',
#     subset='training')  # Use subset='training' for the training set
#
# validation_generator = train_datagen.flow_from_directory(
#     dataset_root,
#     target_size=(sz, sz),
#     batch_size=batch_size,
#     color_mode='grayscale',
#     class_mode='categorical',
#     subset='validation')  # Use subset='validation' for the validation set
#
# # Define the CNN model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(units=96, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=24, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=5,
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator))
#
# # Save the model architecture as JSON and weights as H5
# model_json = model.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# model.save('model-bw.h5')
# print('Model Saved')
#
# test_generator = train_datagen.flow_from_directory(
#     dataset_root,
#     target_size=(sz, sz),
#     batch_size=batch_size,
#     color_mode='grayscale',
#     class_mode='categorical',
#     subset='validation')  # Use subset='validation' for th
#
# test_accuracy = model.evaluate_generator(test_generator, steps=len(test_generator))
#
# print('Test accuracy: {:.2f}%'.format(test_accuracy[1] * 100))
#
