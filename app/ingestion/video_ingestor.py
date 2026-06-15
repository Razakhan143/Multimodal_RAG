from moviepy import VideoFileClip
import json
import os
# Global dictionary to act as source of truth for timestamps
frame_timestamps = {}

def process_video(video_path, output_folder="app\\ingestion\\video_ingestor", audio_output_path="app\\ingestion\\audio_ingestor.py", fps=1):
    global frame_timestamps
    frame_timestamps = {}
    print("🎬 Loading video clip...")
    clip = VideoFileClip(video_path)

    # 1. Extract Frames and track exact time
    print(f"🖼️ Extracting frames at {fps} FPS...")
    duration = clip.duration
    n_frames = int(duration * fps) + 1
    
    for i in range(n_frames):
        t = i / fps
        if t > duration: break
        filename = f"frame{i:04d}.png"
        save_path = os.path.join(output_folder, filename)
        clip.save_frame(save_path, t=t)
        frame_timestamps[filename] = t

    # Save mapping to disk for persistence
    with open(os.path.join(output_folder, 'timestamps.json'), 'w') as f:
        json.dump(frame_timestamps, f)

    # 2. Extract Audio
    print("🎵 Extracting audio track...")
    audio = clip.audio
    audio.write_audiofile(audio_output_path, logger=None)
    print("✅ Video processing complete.")
