import numpy as np
import sounddevice as sd  # for audio playback
import soundfile as sf      # for reading audio files
import threading            # to handle user input concurrently
import matplotlib.pyplot as plt
import librosa               # for audio processing
import time
import mne_lsl
import mne                  # for EEG stream handling
from mne_lsl.lsl import StreamInlet, StreamInfo, resolve_streams
from multiprocessing import Process, Queue, Value  # for parallel EEG processing and shared state

# Suppress verbose MNE logging
mne.set_log_level('CRITICAL')

# Load short audio files for left (speech) and right (music) channels
left_data, fs_left = sf.read('speech.wav')
right_data, fs_right = sf.read('music.wav')

# Ensure mono by selecting first channel if needed
if left_data.ndim > 1:
    left_data = left_data[:, 0]
if right_data.ndim > 1:
    right_data = right_data[:, 0]

# Check that both files have the same sample rate
if fs_left != fs_right:
    raise ValueError("Sample rates of the two files do not match.")

# Load and standardize long speech stimulus, then resample to 500 Hz
long_audio, long_sr = librosa.load(
    '../../Stimuli/Cindy/speech_only_long_22kHz/jane_eyre_05_part1.wav', sr=None
)
long_audio = (long_audio - np.mean(long_audio)) / np.std(long_audio)
long_audio1 = librosa.resample(long_audio, orig_sr=long_sr, target_sr=500)

# Load and standardize long music stimulus, then resample to 500 Hz
long_audio, long_sr = librosa.load(
    '../../Stimuli/Cindy/piano_only_long_22kHz/piano_4_1_22050Hz.wav', sr=None
)
long_audio = (long_audio - np.mean(long_audio)) / np.std(long_audio)
long_audio2 = librosa.resample(long_audio, orig_sr=long_sr, target_sr=500)

# Use the sample rate of the short files for playback
fs = fs_left

# Flatten arrays to ensure 1D
left_data = left_data.flatten()
right_data = right_data.flatten()

# Pad the shorter array so both channels have equal length
max_len = max(len(left_data), len(right_data))
if len(left_data) < max_len:
    left_data = np.pad(left_data, (0, max_len - len(left_data)))
if len(right_data) < max_len:
    right_data = np.pad(right_data, (0, max_len - len(right_data)))

# Stack into stereo data (samples x 2)
data = np.stack((left_data, right_data), axis=-1)

# Shared values for dynamic gain adjustments (range 0.0 to 1.0)
left_gain = Value('d', 0.5)   # initial speech gain
right_gain = Value('d', 0.5)  # initial music gain

# Position index for audio callback
position = 0
lock = threading.Lock()  # protect shared state

def audio_callback(outdata, frames, time, status):
    global position
    with lock:
        start = position
        end = start + frames
        chunk = data[start:end].copy()
        # If at end of data, pad and stop
        if len(chunk) < frames:
            chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)), 'constant')
            raise sd.CallbackStop()

        # Read current gains
        lg = left_gain.value
        rg = right_gain.value

        # Apply gains to each channel and output
        chunk[:, 0] *= lg
        chunk[:, 1] *= rg
        outdata[:] = chunk
        position += frames

def input_thread():
    # Thread to read user balance input from console
    global left_gain, right_gain
    while True:
        try:
            val = float(input("Enter left balance (0.0 - 1.0): "))
            if 0.0 <= val <= 1.0:
                with lock:
                    left_gain.value = val
                    right_gain.value = 1.0 - val
                print(f"Updated balance: Left={left_gain.value:.2f}, Right={right_gain.value:.2f}")
            else:
                print("Please enter a number between 0 and 1.")
        except ValueError:
            print("Invalid input, please enter a decimal number.")

def eeg_thread(q):
    # Thread to stream EEG data, compute gains via CCA, and send to main via queue
    from eeg_cca_from_chunk import eeg_cca_from_chunk

    streams = resolve_streams()
    chunk_duration = 20  # seconds per segment
    fs = 500             # EEG sampling rate after resampling

    # Find the target EEG stream by name
    target_name = 'actiCHamp-24060339'
    stream_idx = None
    for i, stream in enumerate(streams):
        if stream.name == target_name:
            stream_idx = i
            print(f"Found stream '{target_name}' at index {stream_idx}")
            break
    if stream_idx is None:
        raise ValueError(f"Stream with name '{target_name}' not found.")

    # Open LSL inlet for chosen stream
    chosen_stream = streams[stream_idx]
    stream_lsl = StreamInlet(sinfo=chosen_stream)

    print(f"\nRecording for {chunk_duration} seconds...")
    while True:
        # Pull EEG chunk and ensure correct shape
        chunk = stream_lsl.pull_chunk(timeout=chunk_duration, max_samples=chunk_duration*fs)
        eeg_chunk = np.array(chunk[0])
        if eeg_chunk.ndim == 1:
            continue
        elif eeg_chunk.shape[1] == 37:
            print('wut')  # placeholder debug message

        # Compute new gain pair and send to main
        gain = eeg_cca_from_chunk(eeg_chunk, fs, long_audio1, long_audio2, long_sr)
        q.put(gain)

def main():
    # Entry point: start input and EEG threads, then play audio with dynamic balance
    print("Resolving available LSL streams...")
    resolve_streams()

    q = Queue()
    threading.Thread(target=input_thread, daemon=True).start()
    Process(target=eeg_thread, args=(q,), daemon=True).start()

    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=fs):
        print("Playing audio... Adjust balance using EEG or console input...")

        maxTime = int(len(data) / fs * 1000)  # total playback time in ms
        currentTime = 0
        while currentTime < maxTime:
            sd.sleep(1)

            # Check for EEG-based gain updates
            if not q.empty():
                r1, r2 = q.get()
                try:
                    if r1 > r2:
                        left_gain.value = min(left_gain.value + 0.2, 0.8)
                    elif r1 < r2:
                        left_gain.value = max(left_gain.value - 0.2, 0.2)
                    right_gain.value = 1 - left_gain.value
                    print(f"[EEG] Updated balance: Speech={left_gain.value:.2f}, Music={right_gain.value:.2f}")
                except Exception:
                    print(f"[EEG] Invalid gain value")

            currentTime += 1

if __name__ == "__main__":
    main()
