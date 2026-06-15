from moviepy import VideoFileClip
import json
import os
import glob
# Global dictionary to act as source of truth for timestamps

def process_video(video_path, output_folder="app/ingestion/video_ingess", audio_output_path="app/ingestion/audio_ingess/audio.mp3", fps=0.9, max_frames=40):
    global frame_timestamps
    frame_timestamps = {}

    # Ensure output directories exist (fresh checkout / first run).
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)

    # Remove frames from any previously processed video so results don't mix.
    for old in glob.glob(os.path.join(output_folder, "frame*.png")):
        try:
            os.remove(old)
        except OSError:
            pass

    print("🎬 Loading video clip...")
    clip = VideoFileClip(video_path)

    # 1. Extract Frames and track exact time
    print(f"🖼️ Extracting frames at {fps} FPS...")
    duration = clip.duration
    fps_frames = int(duration * fps) + 1

    if fps_frames > max_frames:
        print(f"Sampling {max_frames} frames evenly across {duration:.1f}s "
              f"(fps {fps} would give {fps_frames}).")
        timestamps = [i * duration / max_frames for i in range(max_frames)]
    else:
        print(f"Extracting frames at {fps} FPS ({fps_frames} frames)...")
        timestamps = [i / fps for i in range(fps_frames) if i / fps <= duration]

    for i, t in enumerate(timestamps):
        if t > duration: break
        filename = f"frame{i:04d}.png"
        save_path = os.path.join(output_folder, filename)
        clip.save_frame(save_path, t=t)
        frame_timestamps[filename] = t

    # Save mapping to disk for persistence
    with open(os.path.join(output_folder, 'timestamps.json'), 'w') as f:
        json.dump(frame_timestamps, f)

    # 2. Extract Audio (some videos have no audio track)
    if clip.audio is not None:
        print("🎵 Extracting audio track...")
        clip.audio.write_audiofile(audio_output_path, logger=None)
    else:
        print("🔇 Video has no audio track; skipping transcription.")
        # Remove any stale audio so transcription doesn't pick up an old file.
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)

    clip.close()
    print("✅ Video processing complete.")
    return frame_timestamps