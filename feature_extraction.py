import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_features(file_path, mp3_file, sr=22050, n_mfcc=13, output_dir="features"):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # 1. Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 2. MFCC (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 3. CQT (Constant-Q Transform)
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr)

    # 5. Chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 6. PLP (Perceptual Linear Prediction) - using MFCC with htk=True
    plp = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, htk=True)

    # Extract the base name of the mp3 file (without extension)
    base_filename = os.path.basename(file_path).replace('.mp3', '')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List of features and their names
    features = {
        'Mel Spectrogram': mel_spectrogram_db,
        'MFCC': mfcc,
        'CQT': cqt,
        'Chromagram': chroma,
        'PLP': plp
    }
    
    # Plot and save each feature as an image
    for feature_name, feature_data in features.items():

       # Create a folder for each feature type and use file name and feature name
        feature_folder = os.path.join(output_dir, f"{feature_name.replace(' ', '_')}")
        if not os.path.exists(feature_folder):
            os.makedirs(feature_folder)
            
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(feature_data, x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{feature_name}")
        plt.tight_layout()

        # Save the image as a .png file in the corresponding feature folder
        output_path = os.path.join(feature_folder, f"{base_filename}-{feature_name.replace(' ', '_')}.png")
        plt.savefig(output_path)
        #print(f"Saved {feature_name} as {output_path}")
        plt.close()
        
    return features

def extract_mel_spectrogram(file_path, output_dir="features/ai/Mel_Spectrogram", sr=22050):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # Compute Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Extract base filename
    base_filename = os.path.basename(file_path).replace('.mp3', '')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save Mel Spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram: {base_filename}")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{base_filename}-Mel_Spectrogram.png")
    plt.savefig(output_path)
    print(f"Saved Mel Spectrogram: {output_path}")
    plt.close()

folder_path = "/data/sg2121/fypdataset/dataset/normal_data/ai"
mp3_files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]

# Process each file
for mp3_file in mp3_files:
    file_path = os.path.join(folder_path, mp3_file)
    print(f"Processing {file_path}...")
    extract_mel_spectrogram(file_path)
    #extract_features(file_path, mp3_file, output_dir="features")
    

