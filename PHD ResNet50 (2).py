import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# ==============================
# PARAMETERS
# ==============================
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
train_path = r'D:\TANG GEI KI\PHD Dataset Split\TRAINING (80%)'
valid_path = r'D:\TANG GEI KI\PHD Dataset Split\VALIDATION (10%)'
test_path  = r'D:\TANG GEI KI\PHD Dataset Split\TESTING (10%)'
OUTPUT_DIR = r'D:\TANG GEI KI\output_figures'

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# DATA GENERATORS
# ==============================
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
valid_generator = valid_datagen.flow_from_directory(
    directory=valid_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
test_generator = test_datagen.flow_from_directory(
    directory=test_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
)

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

print(f"Steps per epoch - Train: {STEP_SIZE_TRAIN}, Valid: {STEP_SIZE_VALID}, Test: {STEP_SIZE_TEST}")

# ==============================
# MODEL DEFINITION
# ==============================
resnet = ResNet50(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)
for layer in resnet.layers:
    layer.trainable = False

x = Flatten()(resnet.output)
prediction = Dense(1, activation='sigmoid')(x)
model = Model(inputs=resnet.input, outputs=prediction)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# ==============================
# TRAINING
# ==============================
model_checkpoint = ModelCheckpoint(
    r'D:\TANG GEI KI\64 Adam ResNet50 (Epoch 200).h5',
    save_best_only=True, monitor='val_loss', mode='min', verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=100,
    callbacks=[model_checkpoint]
)

# ==============================
# TEST EVALUATION (Accuracy & Loss)
# ==============================
print("\nEvaluating on test dataset...")
test_loss, test_acc = model.evaluate(test_generator, steps=STEP_SIZE_TEST, verbose=1)
print(f"\n✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Save test results
test_result_path = os.path.join(OUTPUT_DIR, 'test_results.txt')
with open(test_result_path, 'w') as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")

print(f"Test results saved to: {test_result_path}")

# ==============================
# PLOT ACCURACY & LOSS
# ==============================
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_loss_curves.png'))
plt.close()

# ==============================
# EVALUATION
# ==============================
test_generator.reset()
Y_pred_prob = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
Y_pred_class = (Y_pred_prob > 0.5).astype("int32").flatten()
Y_true = test_generator.classes[:len(Y_pred_class)]
target_names = list(train_generator.class_indices.keys())

# Classification report
report = classification_report(Y_true, Y_pred_class, target_names=target_names)
print("\nClassification Report:")
print(report)

# Save report to file
report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"Classification report saved to: {report_path}")

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(Y_true, Y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# ==============================
# ROC CURVE (AUC)
# ==============================
fpr, tpr, _ = roc_curve(Y_true, Y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_auc_curve.png'))
plt.close()

# ==============================
# PRECISION-RECALL CURVE
# ==============================
precision, recall, _ = precision_recall_curve(Y_true, Y_pred_prob)
plt.figure()
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'))
plt.close()

print(f"\n✅ All plots and classification report saved to: {OUTPUT_DIR}")
