import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Target image (256x256 as used in training)
img_path = r"C:\Users\kikit\OneDrive\Documents\UNIMAP PHD\A. TRAIN DATASET SPLIT\TESTING (10%)\MYC+\P2152-15 B192615_patch_88_78.png"
IMG_SIZE = (256, 256)


def get_img_array(img_path, size):
  img = load_img(img_path, target_size=size)
  array = img_to_array(img)
  array = array / 255.0  # Same 1/255 scaling used in ImageDataGenerator
  array = np.expand_dims(array, axis=0)
  return array, img


# Target the last convolutional layer in the respective backbone
# DenseNet121: 'conv5_block16_2_conv'
# MobileNetV2: 'Conv_1' or 'out_relu'
# ResNet50:    'conv5_block3_out'
# VGG16:       'block5_conv3'
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
  # Construct a sub-model mapping input -> last conv layer activations & final output
  last_conv_layer = model.get_layer(last_conv_layer_name)
  grad_model = tf.keras.models.Model(
      inputs=model.inputs, outputs=[last_conv_layer.output, model.output]
  )

  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    # Binary classification target score (predicted MYC probability)
    loss = predictions[:, 0]

  # Compute gradient of output score with respect to feature map activations
  grads = tape.gradient(loss, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  # Weight activation maps by gradient importance
  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  # Apply ReLU and normalize between 0 and 1
  heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
  return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, cam_path="cam.png", alpha=0.4):
  img_np = img_to_array(img)
  heatmap = np.uint8(255 * heatmap)
  jet = plt.get_cmap("jet")
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]
  jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((img_np.shape[1], img_np.shape[0]))
  jet_heatmap = img_to_array(jet_heatmap)

  superimposed_img = jet_heatmap * alpha + img_np * (1 - alpha)
  superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

  plt.figure(figsize=(6, 6))
  plt.imshow(superimposed_img)
  plt.axis("off")
  plt.savefig(cam_path, bbox_inches="tight", pad_inches=0)
  plt.close()


# Example execution for your best-saved model
# model = load_model('best_mobilenetv2_model.h5')
# img_array, original_img = get_img_array(img_path, IMG_SIZE)
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='out_relu')
# save_and_display_gradcam(original_img, heatmap, 'mobilenetv2_gradcam.png')
