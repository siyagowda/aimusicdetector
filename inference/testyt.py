import yt_dlp

def download_mp3(url, output_path='/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/test_dataset'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Usage
url = "https://www.youtube.com/watch?v=tdDFWz7HqZI"
download_mp3(url)