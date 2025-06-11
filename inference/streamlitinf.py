import streamlit as st
from pathlib import Path
import time
import tempfile
import shutil

#from predict import predict
from preprocess import preprocess
from predict import predict

# Constants
FEATURES_DIR = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features")
CHUNKS_DIR = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/temp_chunks")
TEMP_WAV = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/temp_resampled.wav")

def clear_directory(directory):
    if directory.exists() and directory.is_dir():
        for item in directory.rglob("*"):
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
            except Exception as e:
                st.warning(f"Failed to delete {item}: {e}")



def main():
    st.title("🎵 AI-Generated Music Detector")
    st.markdown("Upload a song and I’ll tell you whether it’s **AI-generated** or **human-generated**.")

    # Model mode selector
    model_mode = st.radio("Select Mode", ["Accurate", "Fast"], horizontal=True, help="Fast mode is quicker but (slightly) less accurate.")

    # Upload MP3
    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

    if uploaded_file is not None:
        if st.button("Run Detection"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_input_path = tmp_file.name

            st.audio(temp_input_path, format='audio/mp3')

            # UI elements
            progress = st.progress(0)
            status = st.empty()

            # Progress callback function
            def update_preprocess_progress(val):
                progress.progress(val)

            # Run Preprocessing
            status.text("Preprocessing...")
            start_time = time.time()
            preprocess(temp_input_path, fast=(model_mode == "Fast"), progress_callback=update_preprocess_progress)
            preprocess_time = time.time() - start_time

            # Run Prediction
            status.text("Predicting...")
            start_time = time.time()
            label, confidence = predict(fast=(model_mode == "Fast"))
            predict_time = time.time() - start_time
            progress.progress(100)

            # Display result
            status.text("Done.")
            st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")

            # Timing information
            st.markdown(f"Preprocessing Time: `{preprocess_time:.2f}s`")
            st.markdown(f"Prediction Time: `{predict_time:.2f}s`")
            total_time = preprocess_time + predict_time
            st.markdown(f"\nTotal Time: `{total_time:.2f}s`")

            # Clean up
            if Path(temp_input_path).exists():
              Path(temp_input_path).unlink()
            if Path("temp_chunks").exists():
              shutil.rmtree("temp_chunks")

if __name__ == "__main__":
    main()
