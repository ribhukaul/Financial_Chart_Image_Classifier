# Financial Chart Image Classifier
Financial Chart Image Classifier
This project implements a Convolutional Neural Network (CNN) for classifying financial chart images. The classifier differentiates between two classes: Correction and Uptrend. The implementation uses TensorFlow and Keras for building and training the model.

Requirements
TensorFlow
NumPy
Pandas
Pillow
SciPy

Steps Followed in the Code
# Import Libraries:
The necessary libraries are imported, including TensorFlow, NumPy, Pandas, and image processing libraries from Keras.

# Check for Dependencies:
The code ensures that the Pillow and SciPy libraries are available. If not, it prompts the user to install them.

# Check for Available GPUs:
The code prints the number of available GPUs to ensure TensorFlow can utilize GPU acceleration if available.

# Define Paths to Data:
The paths to the dataset directories are defined. The dataset includes images of financial charts categorized into 'Correction' and 'Uptrend'. The paths to these categories and the labels file are set.

# Load Labels:
The labels are loaded from an Excel file using Pandas.

# Load and Preprocess Data:
The images are loaded and preprocessed. The images are resized to 150x150 pixels. The images are converted to arrays and appended to a list along with their labels.

# Convert Lists to NumPy Arrays:
The lists of images and labels are converted to NumPy arrays for easier manipulation and feeding into the model.

# Normalize Data:
The image data is normalized to have pixel values between 0 and 1.

# Prepare Training Data:
The data is split into two classes and then combined into a single dataset without further splitting.

# Define the CNN Model:
The CNN model is defined using Keras. The model includes:

Input layer
Three convolutional layers with ReLU activation and stride of 2
A flattening layer
A dense layer with 128 units and ReLU activation
An output layer with a sigmoid activation for binary classification
Compile the Model:
The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as the metric.

# Data Augmentation:
Data augmentation is applied to the training data to improve the model's robustness. This includes random zoom, rotation, and horizontal flip.

# Prepare the Data Generator:
A data generator is created to feed the training data into the model. All data is used in each epoch.

# Custom Callback for Printing Metrics:
A custom callback is defined to print the loss and accuracy after each epoch.

# Train the Model:
The model is trained using the data generator, and the custom callback is used to print metrics. Training is done for 20 epochs.

# Preprocess Test Images:
A function is defined to preprocess test images. The images are resized, normalized, and expanded to add a batch dimension.

# Predict on Test Set:
The model is used to predict the class of each image in the test set. The predictions are stored for comparison with the actual labels.

# Compare Predictions with Actual Labels:
The predictions are compared with the actual labels from the test set. The accuracy is calculated based on the number of correct predictions.

# Print Accuracy and Mismatches:
The final accuracy of the model on the test set is printed. Additionally, any mismatches between the predicted and actual labels are printed for further analysis.