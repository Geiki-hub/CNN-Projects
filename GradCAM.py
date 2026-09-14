import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

img_path = r"xxx"
IMG_SIZE = (256, 256)

def get_img_array(image_path, size):
  img = load_img(image_path, target_size=size)
  array = img_to_array(img)
  array = array / 255.0
  array = np.expand_dims(array, axis=0)
  return array, img

def find_target_layer(model, layer_name):
  try:
    return model.get_layer(layer_name)
  except ValueError:
    for layer in model.layers:
      if hasattr(layer, "layers"):
        try:
          return layer.get_layer(layer_name)
        except ValueError:
          continue
  raise ValueError(f"Layer '{layer_name}' not found in the model architecture.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
  target_layer = find_target_layer(model, last_conv_layer_name)
    
  grad_model = tf.keras.models.Model(
      inputs=model.inputs, outputs=[target_layer.output, model.output]
  )

  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

  grads = tape.gradient(loss, conv_outputs)

  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  heatmap = tf.maximum(heatmap, 0.0)

  max_val = tf.math.reduce_max(heatmap)
  if max_val > 1e-8:
    heatmap = heatmap / max_val
  else:
    heatmap = tf.zeros_like(heatmap)

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
  plt.savefig(cam_path, bbox_inches="tight", pad_inches=0, dpi=300)
  plt.close()
  print(f"Saved Grad-CAM overlay to: {cam_path}")
