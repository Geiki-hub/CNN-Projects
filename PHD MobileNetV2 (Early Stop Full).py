import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# --- Paths ---
IMAGE_SIZE = (256, 256)
train_path = r'D:\TANG GEI KI\PHD Dataset Split\TRAINING (80%)'
valid_path = r'D:\TANG GEI KI\PHD Dataset Split\VALIDATION (10%)'
test_path  = r'D:\TANG GEI KI\PHD Dataset Split\TESTING (10%)'

BATCH_SIZE = 64

# --- Data Generators ---
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
valid_generator = valid_datagen.flow_from_directory(
    valid_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
test_generator  = test_datagen.flow_from_directory(
    test_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST  = test_generator.n // test_generator.batch_size

# --- Load MobileNetV2 ---
mobilenet = MobileNetV2(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)

for layer in mobilenet.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(mobilenet.output)
prediction = Dense(1, activation='sigmoid')(x)
model = Model(inputs=mobilenet.input, outputs=prediction)

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# --- Callbacks ---
model_checkpoint = ModelCheckpoint(
    r'D:\TANG GEI KI\PHD Dataset Split\64 Adam MobileNetV2 (early stop).h5',
    save_best_only=True, monitor='val_loss', mode='min', verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
)

# --- Train Model ---
history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=200,
    callbacks=[model_checkpoint, early_stopping]
)

# --- Plot Training & Validation Curves with Early Stopping Marker ---
best_epoch = np.argmin(history.history['val_loss'])

plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(best_epoch, color='r', linestyle='--', label='Early Stopping')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axvline(best_epoch, color='r', linestyle='--', label='Early Stopping')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

# --- Evaluate on Test Set ---
test_loss, test_accuracy = model.evaluate(test_generator, steps=STEP_SIZE_TEST, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# --- Predictions ---
y_pred_probs = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).ravel()
y_true = test_generator.classes[:len(y_pred)]

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --- Classification Report ---
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

# --- Additional Metrics ---
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# --- ROC Curve & AUC ---
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Precision-Recall Curve ---
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_pred_probs)
avg_precision = average_precision_score(y_true, y_pred_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, color='green', lw=2,
         label=f'Precision-Recall (AP = {avg_precision:.4f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
