import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
This file contains code for random forest classifier.
'''

# Load data
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

