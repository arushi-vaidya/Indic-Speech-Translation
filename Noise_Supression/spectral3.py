import os
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import queue
import webrtcvad
from scipy.ndimage import uniform_filter1d

samplerate = 16000
blocksize = 320  # 20 ms frames for VAD at 16kHz(bad conversion over 20 ms maybe due to gpu limitation of my laptop )
output_file = r"enhanced_realtime.wav"
frame_length = 2048
hop_length = 512
alpha = 0.98
noise_threshold = 1.2
q = queue.Queue()
buffer = np.zeros(0)
noise_profile = np.zeros((frame_length // 2 + 1, 1))
vad = webrtcvad.Vad(2)  # goes from 0 to 3 (more strict as it increases) (basically detection sensitivity)
all_enhanced_audio = []


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())


in_stream = sd.InputStream(samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback)
in_stream.start()

print("Recording started with VAD and real-time enhancement. Press Ctrl+C to stop.")

try:
    while True:
        chunk = q.get().flatten()
        buffer = np.append(buffer, chunk)

        if len(buffer) >= frame_length:
            # changing from time domain to frq domain
            stft = librosa.stft(buffer, n_fft=frame_length, hop_length=hop_length)
            mag, phase = np.abs(stft), np.angle(stft)
            avg_mag = np.mean(mag, axis=1, keepdims=True)

            # (must be 20ms for webrtcvad) idk why but it works better with 20ms
            vad_frame = buffer[:blocksize]  # first 20ms frame exactly
            vad_frame_bytes = (vad_frame * 32768).astype(np.int16).tobytes()
            is_speech = vad.is_speech(vad_frame_bytes, samplerate)

            if not is_speech:
                noise_profile = alpha * noise_profile + (1 - alpha) * avg_mag

            # (changed mask for smooth audio) remove this if its still same *****
            mask = mag / (mag + noise_threshold * noise_profile + 1e-8) #**tune**
            mag_enh = mag * mask
            mag_enh = uniform_filter1d(mag_enh, size=3, axis=1)

            # changing it back to time domain
            stft_enh = mag_enh * np.exp(1j * phase)
            enhanced = librosa.istft(stft_enh, hop_length=hop_length)

            # just for testing remove later
            all_enhanced_audio.append(enhanced)

            buffer = np.zeros(0)

except KeyboardInterrupt:
    print("Recording stopped. Saving output...")

    final_audio = np.concatenate(all_enhanced_audio)
    final_audio /= np.max(np.abs(final_audio)) 
    sf.write(output_file, final_audio, samplerate)
    print(f"Enhanced audio saved at:\n{output_file}")
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(final_audio, sr=samplerate, color='r')

    plt.title("Final Enhanced Audio (Real-Time, Adaptive, VAD)")
    plt.tight_layout()
    plt.show()

    in_stream.stop()
