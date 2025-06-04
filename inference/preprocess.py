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
import torchaudio
import torchaudio.transforms as T

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
    
    # Save chunks to disk
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

    # Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    # Loop through audio files
    for audio_path in input_dir.glob("*.*"): 
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

    # 4. Chromagram
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
        print(f"Saved {feature_name} as {output_path}")
        plt.close()
        
    return features

def extract_mfcc_and_melspec(input_path):
    target_sr = 22050
    target_width = 256

    # Output directories
    output_dir = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features")
    mfcc_out_path = output_dir / f"{Path(input_path).stem}_MFCC_tensor.pt"
    mel_out_path = output_dir / f"{Path(input_path).stem}_Mel_Spectrogram_tensor.pt"

    # Load audio
    waveform, sr = torchaudio.load(input_path)

    # Resample if needed
    if sr != target_sr:
        resample = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resample(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # ----- MFCC -----
    mfcc_transform = T.MFCC(
        sample_rate=target_sr,
        n_mfcc=40,
        melkwargs={
            "n_fft": 1024,
            "hop_length": 512,
            "n_mels": 128,
            "center": True,
            "power": 2.0,
        },
    )
    mfcc = mfcc_transform(waveform)
    _, _, t = mfcc.shape
    if t < target_width:
        mfcc = torch.nn.functional.pad(mfcc, (0, target_width - t))
    else:
        start = (t - target_width) // 2
        mfcc = mfcc[:, :, start:start + target_width]
    torch.save(mfcc, mfcc_out_path)

    # ----- Mel Spectrogram -----
    mel_transform = T.MelSpectrogram(sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=128)
    to_db = T.AmplitudeToDB(top_db=80)
    mel = to_db(mel_transform(waveform))
    _, _, t = mel.shape
    if t < target_width:
        mel = torch.nn.functional.pad(mel, (0, target_width - t))
    else:
        start = (t - target_width) // 2
        mel = mel[:, :, start:start + target_width]
    torch.save(mel, mel_out_path)

    print(f"Saved MFCC to {mfcc_out_path}")
    print(f"Saved Mel Spectrogram to {mel_out_path}")


def preprocess(path):
    split_mp3_into_chunks(path)
    chunk_path = Path("temp_chunks")

    for audio_file in chunk_path.glob("*.mp3"):
        resample_and_normalize(str(audio_file), str(audio_file), target_sample_rate=24000)
        extract_features(audio_file)
        extract_mfcc_and_melspec(audio_file)

    #transcribe_lyrics(chunk_path)

