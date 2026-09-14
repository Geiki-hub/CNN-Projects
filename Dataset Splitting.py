import os
import re
import shutil
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

SOURCE_DIR = r"xxx"

OUTPUT_BASE = r"xxx"
TRAIN_DIR = os.path.join(OUTPUT_BASE, "TRAINING (80%)")
VALID_DIR = os.path.join(OUTPUT_BASE, "VALIDATION (10%)")
TEST_DIR = os.path.join(OUTPUT_BASE, "TESTING (10%)")

def extract_slide_id(filename):
  match = re.search(
      r"(WSI_MYC_[a-zA-Z]+_\d+|WSI_\d+|SLIDE_\d+|PATIENT_\d+|[A-Za-z0-9]+(?=_tile))",
      filename,
  )
  return match.group(0) if match else filename.split("_")[0]

slide_to_patches = defaultdict(list)
slide_to_class = {}

class_names = ["MYC_POS", "MYC_NEG"]

for class_name in class_names:
  class_src = os.path.join(SOURCE_DIR, class_name)
  if not os.path.exists(class_src):
    continue

  all_files = [
      f
      for f in os.listdir(class_src)
      if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
  ]

  for fname in all_files:
    sid = extract_slide_id(fname)
    full_src = os.path.join(class_src, fname)
    slide_to_patches[sid].append(full_src)
    slide_to_class[sid] = class_name

unique_slides = np.array(list(slide_to_patches.keys()))
labels = np.array([slide_to_class[sid] for sid in unique_slides])

print(f"Total Unique Slides Identified: {len(unique_slides)}")
print(
    f"Class distribution: {np.sum(labels == 'MYC_POS')} MYC+ |"
    f" {np.sum(labels == 'MYC_NEG')} MYC-"
)

train_slides, temp_slides, train_labels, temp_labels = train_test_split(
    unique_slides, labels, test_size=0.20, stratify=labels, random_state=42
)

val_slides, test_slides, val_labels, test_labels = train_test_split(
    temp_slides, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
)

print(
    f"\nTrain Set: {len(train_slides)} WSIs (MYC+: {np.sum(train_labels == 'MYC_POS')}, MYC-: {np.sum(train_labels == 'MYC_NEG')})"
)
print(
    f"Val Set:   {len(val_slides)} WSIs (MYC+: {np.sum(val_labels == 'MYC_POS')}, MYC-: {np.sum(val_labels == 'MYC_NEG')})"
)
print(
    f"Test Set:  {len(test_slides)} WSIs (MYC+: {np.sum(test_labels == 'MYC_POS')}, MYC-: {np.sum(test_labels == 'MYC_NEG')})"
)

split_mapping = [
    (TRAIN_DIR, train_slides),
    (VALID_DIR, val_slides),
    (TEST_DIR, test_slides),
]

for target_root, slide_list in split_mapping:
  for sid in slide_list:
    c_name = slide_to_class[sid]
    dest_folder = os.path.join(target_root, c_name)
    os.makedirs(dest_folder, exist_ok=True)

    for src_path in slide_to_patches[sid]:
      fname = os.path.basename(src_path)
      shutil.copyfile(src_path, os.path.join(dest_folder, fname))

print("\nDataset partitioning complete.")
