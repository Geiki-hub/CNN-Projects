import os
import re
import shutil
from collections import defaultdict
import numpy as np

SOURCE_DIR = r"xxx"

OUTPUT_BASE = r"xxx"
TRAIN_DIR = os.path.join(OUTPUT_BASE, "TRAINING (80%)")
VALID_DIR = os.path.join(OUTPUT_BASE, "VALIDATION (10%)")
TEST_DIR = os.path.join(OUTPUT_BASE, "TESTING (10%)")

def extract_slide_id(filename):
  match = re.search(r"(WSI_\d+|SLIDE_\d+|PATIENT_\d+|[A-Za-z0-9]+(?=_tile))", filename)
  return match.group(0) if match else filename.split("_")[0]

np.random.seed(42) 

for class_name in ["MYC_POS", "MYC_NEG"]:
  class_src = os.path.join(SOURCE_DIR, class_name)
  if not os.path.exists(class_src):
    continue

  all_files = [
      f
      for f in os.listdir(class_src)
      if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
  ]

  slide_to_patches = defaultdict(list)
  for fname in all_files:
    sid = extract_slide_id(fname)
    slide_to_patches[sid].append(fname)

  unique_slides = np.array(list(slide_to_patches.keys()))
  np.random.shuffle(unique_slides)

  total_slides = len(unique_slides)
  n_test = int(np.round(total_slides * 0.10))
  n_val = int(np.round(total_slides * 0.10)) 
  n_train = total_slides - (n_test + n_val) 

  test_slides = unique_slides[:n_test]
  val_slides = unique_slides[n_test : n_test + n_val]
  train_slides = unique_slides[n_test + n_val :]

  print(f"\n--- Class: {class_name} ---")
  print(f"Total WSIs: {total_slides}")
  print(f"Train WSIs: {len(train_slides)} | Val WSIs: {len(val_slides)} | Test WSIs: {len(test_slides)}")

  split_mapping = [
      (TRAIN_DIR, train_slides),
      (VALID_DIR, val_slides),
      (TEST_DIR, test_slides),
  ]

  for target_root, slide_list in split_mapping:
    dest_folder = os.path.join(target_root, class_name)
    os.makedirs(dest_folder, exist_ok=True)

    for sid in slide_list:
      for fname in slide_to_patches[sid]:
        src_path = os.path.join(class_src, fname)
        dest_path = os.path.join(dest_folder, fname)
        shutil.copyfile(src_path, dest_path)

print("\nDataset partitioning complete.")
