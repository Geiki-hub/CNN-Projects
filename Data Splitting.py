import os
import shutil
import random
from tqdm import tqdm

input_dir = r'D:\TANG GEI KI\PHD Dataset Split\Combined Reactive Lymph Node'

output_dirs = {
    'train': r'D:\TANG GEI KI\PHD Dataset Split\TRAINING (80%)\REACTIVE LYMPH NODE',
    'val':   r'D:\TANG GEI KI\PHD Dataset Split\VALIDATION (10%)\REACTIVE LYMPH NODE',
    'test':  r'D:\TANG GEI KI\PHD Dataset Split\TESTING (10%)\REACTIVE LYMPH NODE'
}

# Create destination folders
for p in output_dirs.values():
    os.makedirs(p, exist_ok=True)

# Collect files
files = [f for f in os.listdir(input_dir)
         if os.path.isfile(os.path.join(input_dir, f))]

print(f"Found {len(files)} files.")
print("Sample files:", files[:5])

if len(files) == 0:
    print("ERROR: No files found in input directory!")
    exit()

# Shuffle
random.shuffle(files)

# Split sizes
total = len(files)
train_end = int(total * 0.8)
val_end = train_end + int(total * 0.1)

train_files = files[:train_end]
val_files = files[train_end:val_end]
test_files = files[val_end:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

def copy_list(file_list, dst):
    for f in tqdm(file_list, desc=f"Copying → {os.path.basename(dst)}"):
        src = os.path.join(input_dir, f)
        dst_path = os.path.join(dst, f)
        shutil.copy(src, dst_path)

copy_list(train_files, output_dirs['train'])
copy_list(val_files, output_dirs['val'])
copy_list(test_files, output_dirs['test'])

print("DONE!")
