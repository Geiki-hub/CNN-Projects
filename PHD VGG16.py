import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint

# Set parameters
IMAGE_SIZE = (256, 256)  # Resize images to 224x224
train_path = r'C:\Users\kikit\OneDrive\Documents\UNIMAP PHD\2. TRAIN DATASET SPLIT\TRAINING (80%)'  # Set the path to your training data
valid_path = r'C:\Users\kikit\OneDrive\Documents\UNIMAP PHD\2. TRAIN DATASET SPLIT\VALIDATION (10%)'  # Set the path to your validation data
test_path = r'C:\Users\kikit\OneDrive\Documents\UNIMAP PHD\2. TRAIN DATASET SPLIT\TESTING (10%)'
# Batch size
BATCH_SIZE = 64

# Initialize ImageDataGenerators for rescaling images
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training, validation, and test datasets
train_generator = train_datagen.flow_from_directory(directory=train_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary")
valid_generator = valid_datagen.flow_from_directory(directory=valid_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary")
test_generator = test_datagen.flow_from_directory(directory=test_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary")

# Calculate steps per epoch for training and validation
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

print(f"Steps per epoch for training: {STEP_SIZE_TRAIN}")
print(f"Steps per epoch for validation: {STEP_SIZE_VALID}")
print(f"Steps per epoch for testing: {STEP_SIZE_TEST}")

# Load VGG16 pre-trained model without the top layer
vgg = VGG16(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)

# Freeze pre-trained layers
for layer in vgg.layers:
    layer.trainable = False

# Add custom layers for binary classification
x = Flatten()(vgg.output)
prediction = Dense(1, activation='sigmoid')(x)

# Create the model object
model = Model(inputs=vgg.input, outputs=prediction)

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Set up ModelCheckpoint callback to save the best model based on validation loss
model_checkpoint = ModelCheckpoint(r'C:\Users\kikit\OneDrive\Documents\UNIMAP PHD\2. TRAIN DATASET SPLIT\64 Adam VGG16 (Epoch 30).h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=30,  # Set number of epochs as required
    callbacks=[model_checkpoint]
)

# Plot Training and Validation Accuracy and Loss over epochs
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

# Evaluate the model on the test set
train_generator.reset()
loss, acc = model.evaluate(train_generator, steps=STEP_SIZE_TRAIN, verbose=1)
print(f"\nTrain-set loss: {loss:.4f}")
print(f"Train-set accuracy: {acc:.2%}")

# Evaluate the model on the test set
valid_generator.reset()
loss, acc = model.evaluate(valid_generator, steps=STEP_SIZE_VALID, verbose=1)
print(f"\nValid-set loss: {loss:.4f}")
print(f"Valid-set accuracy: {acc:.2%}")

# Evaluate the model on the test set
test_generator.reset()
loss, acc = model.evaluate(test_generator, steps=STEP_SIZE_TEST, verbose=1)
print(f"\nTest-set loss: {loss:.4f}")
print(f"Test-set accuracy: {acc:.2%}")