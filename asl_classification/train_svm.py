import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
# Assuming you have a dataset with features (86-length landmark arrays) and corresponding labels (1 of 26 alphabets)
# Replace X_train, y_train, X_test, y_test with your actual data

# Assuming X_train and X_test are arrays with shape (number_of_samples, 86)
# and y_train, y_test are arrays with shape (number_of_samples,)
# Example:
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

'''
This file trains a svm model and prints it's accuracy.
'''
data_dict = pickle.load(open('data.pickle', 'rb'))
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
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the SVM classifier
classifier = SVC(kernel='linear', C=1.0, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

f = open('svm_model.p', 'wb')
pickle.dump({'model': classifier}, f)
f.close()
