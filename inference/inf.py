import sys
from pathlib import Path
import time

from predict import predict
from preprocess import preprocess

def clear_directory(directory):
    if directory.exists() and directory.is_dir():
        for item in directory.rglob("*"):  # recursive
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
            except Exception as e:
                print(f"Failed to delete {item}: {e}")

if __name__ == "__main__":
    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features"))
    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/temp_chunks"))

    n = len(sys.argv)
    if n != 2:
      print("Incorrect number of arguments")

    # Arguments passed
    print("\nPath to song:", sys.argv[1])
    song_path = sys.argv[1]

    model = input(print("Accurate or Fast?:"))
    elapsed1 = 0
    elapsed2 = 0

    if model == "Fast":

        print("Preprocessing ...")
    
        start_time = time.time()
        preprocess(song_path, fast=True)
        end_time = time.time()
        elapsed1 = end_time - start_time
        print(f"\nPreprocessing complete in {elapsed1:.2f} seconds.")

        start_time = time.time()
        predict(fast=True)
        end_time = time.time()
        elapsed2 = end_time - start_time
        print(f"\nPredicting complete in {elapsed2:.2f} seconds.")
    
    else:

        print("Preprocessing ...")
    
        start_time = time.time()
        preprocess(song_path)
        end_time = time.time()
        elapsed1 = end_time - start_time
        print(f"\nPreprocessing complete in {elapsed1:.2f} seconds.")

        start_time = time.time()
        predict()
        end_time = time.time()
        elapsed2 = end_time - start_time
        print(f"\nPredicting complete in {elapsed2:.2f} seconds.")


    total = elapsed1 + elapsed2
    print(f"\nTotal inference time: {total:.2f} seconds.")

    # Delete all files from input
    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features"))
    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/temp_chunks"))

    file_path = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/temp_resampled.wav")
    if file_path.exists():
        file_path.unlink()
