import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure PIL is available
try:
    from PIL import Image
except ImportError as e:
    print(f"PIL import error: {e}")
    print("Please install Pillow: pip install Pillow")
    exit()

# Ensure scipy is available
try:
    import scipy
except ImportError as e:
    print(f"scipy import error: {e}")
    print("Please install scipy: pip install scipy")
    exit()

# Check for available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Paths to the data
dataset_dir = 'D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/TimeSeriesPattern_CompVisionModel/WithCandles/stock_chart_images'
correction_dir = os.path.join(dataset_dir, 'Correction')
uptrend_dir = os.path.join(dataset_dir, 'Uptrend')
train_labels_path = os.path.join(dataset_dir, 'trainingset_updated.xlsx')

# Load the labels
df = pd.read_excel(train_labels_path)

# Load and preprocess the data
data = []
labels = []

for index, row in df.iterrows():
    img_name = row['data']
    label = row['class']

    # Determine the folder based on the class
    if label == 0:
        folder = 'Correction'
    else:
        folder = 'Uptrend'

    img_path = os.path.join(dataset_dir, folder, img_name)
    if os.path.exists(img_path):
        try:
            img_array = img_to_array(load_img(img_path, target_size=(150, 150)))  # Adjust the target size as needed
            data.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Convert lists to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"Total images: {len(data)}")
print(f"Total labels: {len(labels)}")

# Split data into classes
data_class_0 = []
data_class_1 = []

for i in range(len(data)):
    if labels[i] == 0:
        data_class_0.append((data[i], labels[i]))
    else:
        data_class_1.append((data[i], labels[i]))

# Combine data into a single dataset without splitting
X_train = np.array([item[0] for item in data_class_0 + data_class_1])
y_train = np.array([item[1] for item in data_class_0 + data_class_1])

print(f"Training data: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")

# Define input shape
input_shape = (150, 150, 3)

# Define the input layer
inputs = Input(shape=input_shape)

# Convolutional layers with stride instead of max pooling
x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2))(inputs)
x = Conv2D(64, (3, 3), activation='relu', strides=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', strides=(2, 2))(x)

# Flatten layer
x = Flatten()(x)

# Dense layers
x = Dense(128, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    rotation_range=10  # Rotating within +/- 10 degrees
)

# Prepare the generator
batch_size = len(X_train)  # Using all data in each epoch
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

# Custom callback to print loss and accuracy after each epoch
class PrintMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")
        print(f"Training - loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}")

# Train the model with callback
history = model.fit(
    train_generator,
    steps_per_epoch=1,  # All data in one step
    epochs=20,  # Adjust epochs as needed
    callbacks=[PrintMetrics()]
)

# Function to preprocess test images
def preprocess_image(img_path, target_size=(150, 150)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction on Test Set
predictions = []

# Path to the directory containing test images
test_dir = 'D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/TimeSeriesPattern_CompVisionModel/WithCandles/stock_chart_images/Test_Set'

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img_array = preprocess_image(img_path)

    try:
        # Predict using the model
        pred = model.predict(img_array)[0]  # Output is a single prediction

        # Convert prediction to class (assuming binary classification)
        predicted_class = 1 if pred > 0.5 else 0

        # Append to predictions list
        predictions.append((img_name, predicted_class))

    except Exception as e:
        print(f"Error predicting for image {img_name}: {str(e)}")

# Convert predictions list to numpy array
predictions = np.array(predictions)

# Print the results (optional)
for img_name, pred_class in predictions:
    print(f"Image: {img_name}, Predicted Class: {pred_class}")

# Store the results in a variable for further processing
results = predictions

# Load testset.xlsx
testset_path = 'D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/TimeSeriesPattern_CompVisionModel/WithCandles/stock_chart_images/testset.xlsx'  # Adjust the path as necessary
df_testset = pd.read_excel(testset_path)

# Ensure the data types are consistent
df_testset['data'] = df_testset['data'].astype(str)
df_testset['class'] = df_testset['class'].astype(int)

# Convert predictions to a dictionary for fast lookup
predicted_dict = {img_name: int(pred_class) for img_name, pred_class in predictions}

# Initialize variables to calculate accuracy
total_entries = len(df_testset)
score = 0

# Iterate through testset.xlsx and compare with predictions
for index, row in df_testset.iterrows():
    img_name = row['data']
    actual_label = row['class']

    if img_name in predicted_dict:
        predicted_label = predicted_dict[img_name]
        if actual_label == predicted_label:
            score += 1
        else:
            print(f"Mismatch: {img_name} - Actual: {actual_label}, Predicted: {predicted_label}")

# Calculate accuracy
accuracy = score / total_entries * 100  # Multiply by 100 to get percentage

print(f"Accuracy: {accuracy:.2f}%")
