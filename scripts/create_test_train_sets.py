import os
import shutil
from pathlib import Path

def split_dataset(
    input_folder='/Users/tom/Code/Bat-O-Meter/src/frame_txt_outputs',
    dest_folder='/Users/tom/Code/Bat-O-Meter/src/yolo/train',
    split_ratio=0.2,
):
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Get all txt files and corresponding png files
    txt_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.txt')])
    png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

    # Ensure matching txt and png files
    frame_numbers = sorted([f.split('_')[1].split('.')[0] for f in txt_files])
    png_frame_numbers = sorted([f.split('_')[1].split('.')[0] for f in png_files])
    assert frame_numbers == png_frame_numbers, "Mismatch between txt and png files"

    # Calculate the number of files to move
    num_files_to_move = int(len(txt_files) * split_ratio)

    for i in range(1, num_files_to_move + 1):
        file_name = f"frame_{i}"

        # Move txt and png files
        shutil.move(os.path.join(input_folder, f"{file_name}.txt"), os.path.join(dest_folder, f"{file_name}.txt"))
        shutil.move(os.path.join(input_folder, f"{file_name}.png"), os.path.join(dest_folder, f"{file_name}.png"))

    print(f"Moved {num_files_to_move} txt and png files to {dest_folder}")
    
    
def do():
    # Adjust this to your actual dataset root
    DATASET_DIR = Path("/Users/tom/Code/Bat-O-Meter/src/yolo")

    # Source folders
    source_folders = {
        'train': DATASET_DIR / 'train',
        'val': DATASET_DIR / 'test',  # assuming "test" is used for validation
    }

    # Destination folders
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Move files
    for split, folder in source_folders.items():
        for file in folder.glob("*"):
            if file.suffix == '.png':
                dest = DATASET_DIR / 'images' / split / file.name
            elif file.suffix == '.txt':
                dest = DATASET_DIR / 'labels' / split / file.name
            else:
                continue  # skip anything not .png or .txt
            shutil.copy2(file, dest)  # use .move if you want to delete originals

    print("✅ Dataset reorganized successfully!")

if __name__ == '__main__':
    do()