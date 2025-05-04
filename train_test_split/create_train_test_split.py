import os
import random

def split_data(input_dirs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure that the split ratios add up to 1
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    
    # Initialize lists to hold file names
    all_train_files = []
    all_val_files = []
    all_test_files = []
    
    # Loop through each input directory (human, ai/segments)
    for input_dir in input_dirs:
        # Get all the files in the current directory
        all_files = os.listdir(input_dir)
        
        # Shuffle the file list
        random.shuffle(all_files)
        
        # Calculate split sizes
        total_files = len(all_files)
        train_size = int(train_ratio * total_files)
        val_size = int(val_ratio * total_files)
        test_size = total_files - train_size - val_size  # Ensure all files are used
        
        # Create the splits for this directory
        train_files = all_files[:train_size]
        val_files = all_files[train_size:train_size + val_size]
        test_files = all_files[train_size + val_size:]
        
        # Append the files to the overall lists
        all_train_files.extend([os.path.join(input_dir, file) for file in train_files])
        all_val_files.extend([os.path.join(input_dir, file) for file in val_files])
        all_test_files.extend([os.path.join(input_dir, file) for file in test_files])

    # Write the file names to text files
    with open('train_files_large.txt', 'w') as f:
        for file in all_train_files:
            f.write(f"{file}\n")

    with open('val_files_large.txt', 'w') as f:
        for file in all_val_files:
            f.write(f"{file}\n")

    with open('test_files_large.txt', 'w') as f:
        for file in all_test_files:
            f.write(f"{file}\n")

    print("Data split completed. Files saved as train_files.txt, val_files.txt, and test_files.txt")

# Example usage
input_dirs = ['/data/sg2121/fypdataset/dataset_large/normal_data/human', '/data/sg2121/fypdataset/dataset_large/normal_data/ai_segments']  # Directories to process
split_data(input_dirs)
