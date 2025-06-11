import streamlit as st
import yt_dlp
import os

# Default output path
DEFAULT_OUTPUT_PATH = '/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/test_dataset'

def download_mp3(url, output_path=DEFAULT_OUTPUT_PATH):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,  # Suppress verbose console output
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return f"{info['title']}.mp3"
    except Exception as e:
        return str(e)

# Streamlit UI
st.title("YouTube MP3 Downloader")

url = st.text_input("Enter YouTube URL")

if st.button("Download MP3"):
    if url:
        with st.spinner("Downloading..."):
            filename = download_mp3(url)
        if filename.endswith(".mp3"):
            st.success(f"Download completed: {filename}")
        else:
            st.error(f"Error: {filename}")
    else:
        st.warning("Please enter a valid URL.")
