import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

slides_myc_pos = [f"WSI_MYC_pos_{i:03d}" for i in range(1, 61)]
slides_myc_neg = [f"WSI_MYC_neg_{i:03d}" for i in range(1, 61)]

all_slides = slides_myc_pos + slides_myc_neg
all_labels = [1] * len(slides_myc_pos) + [0] * len(slides_myc_neg)

train_slides, temp_slides, train_labels, temp_labels = train_test_split(
    all_slides, all_labels, test_size=0.20, stratify=all_labels, random_state=42
)

val_slides, test_slides, val_labels, test_labels = train_test_split(
    temp_slides, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
)

print(f"Total Cohort: {len(all_slides)} WSIs")
print(f"Train Set:    {len(train_slides)} WSIs (MYC+: {sum(train_labels)}, MYC-: {len(train_labels)-sum(train_labels)})")
print(f"Val Set:      {len(val_slides)} WSIs (MYC+: {sum(val_labels)}, MYC-: {len(val_labels)-sum(val_labels)})")
print(f"Test Set:     {len(test_slides)} WSIs (MYC+: {sum(test_labels)}, MYC-: {len(test_labels)-sum(test_labels)})")
