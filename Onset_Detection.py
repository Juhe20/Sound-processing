import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks, correlate
import sounddevice as sd

filename = "disco.00000.wav"
audio, sr = sf.read(filename)
time = np.arange(len(audio)) / sr

#Full-wave rectify signal
rectified = np.abs(audio)

#Low-pass filter
win_size = int(0.02 * sr)
smoothed = np.convolve(rectified, np.ones(win_size) / win_size, mode='same')

plt.figure(figsize=(10, 3))
plt.plot(time, rectified, label='Rectified')
plt.plot(time, smoothed, label='Smoothed', linewidth=2)
plt.title("Smoothed vs. Rectified Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

#Onset detection function
odf = np.diff(smoothed, prepend=smoothed[0])
energy_difference = odf / np.max(odf)

#Pick peaks
threshold = 0.05
min_distance = int(0.1 * sr)
peaks, _ = find_peaks(energy_difference, height=threshold, distance=min_distance)

#Plot of onset detection function with peaks
plt.figure(figsize=(10, 3))
plt.plot(time, energy_difference, label="Energy difference")
plt.plot(time[peaks], energy_difference[peaks], "rx", label="Onsets")
plt.title("Onset Detection Function")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Energy Change")
plt.legend()
plt.tight_layout()
plt.show()

#Autocorrelation
autocorrelation = correlate(energy_difference, energy_difference, mode="full")
autocorrelation = autocorrelation[len(autocorrelation)//2:]
delay = np.arange(len(autocorrelation)) / sr

#min/max delay making sure the tempo is within reasonable music tempo range.
min_delay = int(sr * 0.3)
max_delay = int(sr * 1.0)
autocorrelation_range = autocorrelation[min_delay:max_delay]
delay_range = delay[min_delay:max_delay]

#Find peak in autocorrelation for tempo
peak_delay = delay_range[np.argmax(autocorrelation_range)]
bpm = 60 / peak_delay

sd.play(audio, sr)
sd.wait()
