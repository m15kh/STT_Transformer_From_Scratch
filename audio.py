import librosa
import torch

waveform, sr = librosa.load('1.wav', sr = 16000)
print(waveform.shape)
print(type(waveform))
print(sr)

x = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)  

print(type(x))

print(x.shape)