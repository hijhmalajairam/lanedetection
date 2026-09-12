import os
import urllib.request

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        urllib.request.urlretrieve(url, dest)
        print("Done.")
    else:
        print(f"{dest} already exists.")

if __name__ == "__main__":
    os.makedirs("test_videos", exist_ok=True)
    videos = {
        "project_video.mp4": "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/project_video.mp4",
        "challenge_video.mp4": "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/challenge_video.mp4"
    }
    for name, url in videos.items():
        download_file(url, os.path.join("test_videos", name))
