import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(
    "Num GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU"))
)

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 200

train_path = (
    r"xxx"
)
valid_path = (
    r"xxx"
)
test_path = (
    r"xxx"
)

checkpoint_dir = r"xxx"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(
    checkpoint_dir, "64 Adam MobileNetV2 (Epoch 200).h5"
)

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42,
)

valid_generator = valid_datagen.flow_from_directory(
    directory=valid_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

STEP_SIZE_TRAIN = int(np.ceil(train_generator.n / BATCH_SIZE))
STEP_SIZE_VALID = int(np.ceil(valid_generator.n / BATCH_SIZE))
STEP_SIZE_TEST = int(np.ceil(test_generator.n / BATCH_SIZE))

print(f"Total Test Patches: {test_generator.n}")

base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), weights="imagenet", include_top=False
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
prediction = Dense(1, activation="sigmoid")(
    x
)

model = Model(inputs=base_model.input, outputs=prediction)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(
    loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

model_checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS,
    callbacks=[model_checkpoint],
)

print(f"\nLoading best checkpoint weights: {best_model_path}")
model.load_weights(best_model_path)

test_generator.reset()
raw_preds = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)

y_probs = raw_preds.ravel()[: test_generator.n]
y_true = test_generator.classes[: len(y_probs)]

np.savez_compressed(
    "MobileNetV2_test_predictions.npz",
    y_true=y_true,
    y_probs=y_probs,
    filenames=test_generator.filenames[: len(y_probs)],
)
print("Saved predictions to 'MobileNetV2_test_predictions.npz'")

tau = 0.5
y_pred = (y_probs >= tau).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
mcc = matthews_corrcoef(y_true, y_pred)
auroc = roc_auc_score(y_true, y_probs)

print("\n" + "=" * 50)
print("MOBILENETV2 TEST METRICS")
print("=" * 50)
print(f"Accuracy:        {accuracy * 100:.2f}%")
print(f"MYC+ Sens (TPR): {sensitivity * 100:.2f}%")
print(f"Specificity:     {specificity * 100:.2f}%")
print(f"PPV (Precision): {ppv * 100:.2f}%")
print(f"NPV:             {npv * 100:.2f}%")
print(f"MCC:             {mcc:.4f}")
print(f"AUROC:           {auroc:.4f}")
print("=" * 50)

print(
    "\nRunning WSI-level cluster bootstrap (B=1,000) for confidence"
    " intervals..."
)

def extract_slide_id(filepath):
  basename = os.path.basename(filepath)
  match = re.search(r"(WSI_\d+|SLIDE_\d+|[A-Za-z0-9]+(?=_patch))", basename)
  return match.group(0) if match else basename.split("_")[0]

filenames = np.array(test_generator.filenames[: len(y_probs)])
slide_ids = np.array([extract_slide_id(f) for f in filenames])
unique_slides = np.unique(slide_ids)
n_slides = len(unique_slides)

print(f"Detected {n_slides} unique WSI clusters in test set.")

B = 1000
boot_acc = []
boot_mcc = []
boot_sens = []

np.random.seed(42)
for b in range(B):
  sampled_slides = np.random.choice(unique_slides, size=n_slides, replace=True)
  sample_indices = np.concatenate(
      [np.where(slide_ids == s)[0] for s in sampled_slides]
  )

  b_true = y_true[sample_indices]
  b_pred = y_pred[sample_indices]

  b_tn, b_fp, b_fn, b_tp = confusion_matrix(
      b_true, b_pred, labels=[0, 1]
  ).ravel()

  boot_acc.append((b_tp + b_tn) / len(b_true))
  boot_mcc.append(matthews_corrcoef(b_true, b_pred))
  boot_sens.append(b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0)

ci_acc = np.percentile(boot_acc, [2.5, 97.5])
ci_mcc = np.percentile(boot_mcc, [2.5, 97.5])
ci_sens = np.percentile(boot_sens, [2.5, 97.5])

print(
    f"Cluster Bootstrapped 95% CI for Accuracy: {ci_acc[0]*100:.2f}% -"
    f" {ci_acc[1]*100:.2f}%"
)
print(f"Cluster Bootstrapped 95% CI for MCC:      {ci_mcc[0]:.4f} - {ci_mcc[1]:.4f}")
print(
    f"Cluster Bootstrapped 95% CI for MYC+ Sens: {ci_sens[0]*100:.2f}% -"
    f" {ci_sens[1]*100:.2f}%"
)
