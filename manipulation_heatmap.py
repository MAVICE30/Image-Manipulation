import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the path to the dataset
path = '/Users/anantkumar/Desktop/Problem Statement/CASIA2.0_revised/'

# Load the dataset
def load_data():
    X = []
    y = []
    for folder in ['Au', 'Tp']:
        for file in os.listdir(path + folder):
            img = cv2.imread(os.path.join(path, folder, file))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                X.append(img)
                if folder == 'Au':
                    y.append(0)
                else:
                    y.append(1)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Split the dataset into training and testing sets
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Load an image to detect manipulation
img = cv2.imread('image.jpg')
img = cv2.resize(img, (224, 224))

# Generate the heatmap
heatmap = model.predict(img[np.newaxis, ...].astype(np.float32))
heatmap = np.squeeze(heatmap)

# Plot the heatmap
sns.heatmap(heatmap, cmap='jet')

# Threshold the heatmap to identify manipulated regions
threshold = 0.5
mask = heatmap > threshold

# Apply the mask to the original image
img_with_box = img.copy()
img_with_box[mask] = [0, 0, 255]

# Plot the original image with bounding box
plt.imshow(img_with_box)
