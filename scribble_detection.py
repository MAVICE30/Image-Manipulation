import os
import cv2
import numpy as np
import pytesseract
import re
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to extract features from an image region
def extract_features(roi):
    return np.mean(roi)

# Define the path to the CASIA dataset
casia_dataset_path = '/path/to/casia_dataset/'

# Create a list to store preprocessed image data and labels
image_data = []
labels = []

# Define the image size to resize to
image_size = (128, 128)

# Load the clean text images
clean_text_dir = os.path.join(casia_dataset_path, 'clean_text')
for filename in os.listdir(clean_text_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(clean_text_dir, filename)
        img = cv2.imread(img_path)

        # Preprocess the image
        img = cv2.resize(img, image_size)  # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
        img = img / 255.0  # Normalization

        image_data.append(img)
        labels.append(0)  # 0 for clean text

# Load the overwritten text images
overwritten_text_dir = os.path.join(casia_dataset_path, 'overwritten_text')
for filename in os.listdir(overwritten_text_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(overwritten_text_dir, filename)
        img = cv2.imread(img_path)

        # Preprocess the image
        img = cv2.resize(img, image_size)  # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
        img = img / 255.0  # Normalization

        image_data.append(img)
        labels.append(1)  # 1 for overwritten text

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Initialize and train a machine learning model (e.g., RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Load the image you want to perform scribbling and overwriting detection on
input_image_path = 'sample_data/scribble.jpg'
image = cv2.imread(input_image_path)

# Perform text detection
detection_results = pytesseract.image_to_boxes(image)

# Initialize a list to store regions with potential issues
potential_issue_regions = []

for detection in detection_results.splitlines():
    detection = detection.split()
    x, y, w, h = int(detection[1]), int(detection[2]), int(detection[3]), int(detection[4])
    roi = image[y:h, x:w]

    # Preprocess the image region similarly to the training data
    roi = cv2.resize(roi, image_size)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = roi / 255.0

    # Extract features from the region
    features = extract_features(roi)

    # Use the trained ML model to predict if the region is overwritten
    prediction = model.predict([features])

    # For demonstration, we'll assume a prediction of 1 indicates overwriting
    if prediction == 1:
        # If the model predicts overwriting, mark the region as a potential issue
        potential_issue_regions.append((x, y, w, h))

# Highlight regions with potential issues
for x, y, w, h in potential_issue_regions:
    cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)

# Display the result
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
