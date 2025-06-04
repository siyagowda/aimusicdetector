import sys
from pydub import AudioSegment
import os
import whisper
import torch
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import shutil

def split_mp3_into_chunks(file_path, chunk_length_sec=30):
    audio = AudioSegment.from_mp3(file_path)
    chunk_length_ms = chunk_length_sec * 1000
    total_length = len(audio)

    if total_length < chunk_length_ms:
        return "song length too short"
    
    # Discard any excess that doesn't fit into a full chunk
    max_valid_length = total_length - (total_length % chunk_length_ms)
    audio = audio[:max_valid_length]
    
    chunks = [
        audio[i:i + chunk_length_ms]
        for i in range(0, len(audio), chunk_length_ms)
    ]
    
    # Optional: save chunks to disk
    output_dir = f"temp_chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, chunk in enumerate(chunks):
        out_file = os.path.join(output_dir, f"temp_part{idx+1}.mp3")
        chunk.export(out_file, format="mp3")
        print(f"Saved {out_file}")

    return chunks

def resample_and_normalize(input_file, output_file, target_sample_rate=24000):
    # Load the MP3 file using pydub
    audio = AudioSegment.from_mp3(input_file)
    
    # Resample the audio to the target sample rate
    audio_resampled = audio.set_frame_rate(target_sample_rate)

    # Export the resampled audio to a temporary WAV file
    temp_wav = "temp_resampled.wav"
    audio_resampled.export(temp_wav, format="wav")

    # Load the WAV file using librosa for normalization
    y, sr = librosa.load(temp_wav, sr=target_sample_rate)
    
    # Normalize the audio signal to be between -1 and 1
    y_normalized = librosa.util.normalize(y)

    # Save normalized WAV using soundfile
    sf.write(temp_wav, y_normalized, sr)

    # Convert back to MP3 using pydub
    audio_normalized = AudioSegment.from_wav(temp_wav)
    audio_normalized.export(output_file, format="mp3")

    print(f"Resampled and normalized audio saved to {output_file}")

def transcribe_lyrics(path):
    input_dir = path
    output_dir = Path("temp_chunks")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Whisper model (auto uses GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    # Loop through audio files
    for audio_path in input_dir.glob("*.*"):  # supports .wav, .mp3, etc.
        output_path = output_dir / f"{audio_path.stem}_lyrics.txt"

        print(f"Transcribing: {audio_path.name}")
        try:
            result = model.transcribe(str(audio_path), fp16=True)
            text = result["text"].strip()

            # Save transcript
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved: {audio_path.name}")

        except Exception as e:
            print(f"Failed to transcribe {audio_path.name}: {e}")

def extract_mel_spectrogram(file_path, output_dir="features/Mel_Spectrogram", sr=22050):
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

def extract_features(file_path, sr=22050, n_mfcc=13, output_dir="features"):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # 1. Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 2. MFCC (Mel-Frequency Cepstral
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 3. CQT (Constant-Q Transform)
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr)

    # 5. Chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Extract the base name of the mp3 file 
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

def preprocess(path):
    split_mp3_into_chunks(path)
    chunk_path = Path("temp_chunks")

    for audio_file in chunk_path.glob("*.mp3"):
        resample_and_normalize(str(audio_file), str(audio_file), target_sample_rate=24000)
        extract_mel_spectrogram(audio_file)

    #transcribe_lyrics(chunk_path)

def get_model():
    # Load the pre-trained ResNet-18 model
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Modify the last layer of the model
    num_classes = 2 # number of classes in dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/mel-spec/cur_model.pt"))
    return model

def predict():
    model = get_model()
    model.eval()

    mel_dir = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel_Spectrogram")
    mel_paths = list(mel_dir.glob("*.png"))  # Or "**/*.png" for recursive search

      
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = []

    for mel_path in mel_paths:
        # Load image and apply transform
        image = Image.open(mel_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            prob = F.softmax(output, dim=1)  # Get probabilities
            predictions.append(prob.cpu())

    print(len(predictions))
    # Average probabilities across all chunks
    avg_prob = torch.mean(torch.cat(predictions, dim=0), dim=0)
    predicted_label = torch.argmax(avg_prob).item()
    confidence = avg_prob[predicted_label].item()

    label_name = "AI-generated" if predicted_label == 0 else "Human-generated"
    print(f"\nPrediction: {label_name} (Confidence: {confidence:.2f})")

    return label_name, confidence

def clear_directory(directory):
    if directory.exists() and directory.is_dir():
        for item in directory.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                #print(f"Deleted: {item}")
            except Exception as e:
                print(f"Failed to delete {item}: {e}")

if __name__ == "__main__":
    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel_Spectrogram"))
    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/temp_chunks"))

    start_time = time.time()

    n = len(sys.argv)
    if n != 2:
      print("Incorrect number of arguments")

    # Arguments passed
    print("\nPath to song:", sys.argv[1])
    song_path = sys.argv[1]

    print("Preprocessing ...")
    preprocess(song_path)

    end_time = time.time()
    elapsed1 = end_time - start_time
    print(f"\n✅ Preprocessing complete in {elapsed1:.2f} seconds.")

    start_time = time.time()
    predict()
    end_time = time.time()
    elapsed2 = end_time - start_time
    print(f"\n✅ Predicting complete in {elapsed2:.2f} seconds.")

    total = elapsed1 + elapsed2
    print(f"\n✅ Total inference time: {total:.2f} seconds.")

    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel_Spectrogram"))
    clear_directory(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/temp_chunks"))