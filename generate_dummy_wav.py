import os, wave, struct

def create_silence(path, duration_sec=1, framerate=16000):
    n_frames = int(duration_sec * framerate)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(framerate)
        silence = struct.pack('<h', 0) * n_frames
        wf.writeframes(silence)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'release_in_the_wild'))
os.makedirs(base_dir, exist_ok=True)
create_silence(os.path.join(base_dir, 'sample1.wav'))
create_silence(os.path.join(base_dir, 'sample2.wav'))
print('Dummy wav files created.')
