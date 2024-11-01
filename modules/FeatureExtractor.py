import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self, n_fft=512, n_mfcc=13, n_mels=40, sample_rate=48000):
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mfcc_transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': n_fft, "hop_length": int(150 * (1 - 0.1) * sample_rate / 1000), "n_mels": n_mels}
        ).to(self.device)
        self.mfcc_features = None

        self.melspectrogram_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=int(150 * (1 - 0.1) * sample_rate / 1000),
            n_mels=n_mels
        ).to(self.device)
        self.melspectrogram_features = None
    
    def extract_mfcc(self, frames, batch_size=60000, normalize=True):
        num_frames = frames.shape[0]
        n_mfcc = self.mfcc_transform.n_mfcc  
        
        mfcc_tensor = torch.zeros((num_frames, n_mfcc), device=self.device)
        
        for i in range(0, num_frames, batch_size):
            batch_frames = frames[i:i + batch_size].to(self.device)  
            mfcc_batch = self.mfcc_transform(batch_frames).mean(dim=-1)  
            mfcc_tensor[i:i + mfcc_batch.shape[0]] = mfcc_batch 

        if not normalize:
            self.mfcc_features = mfcc_tensor
            return mfcc_tensor 
        
        mfcc_min = mfcc_tensor.min(dim=0, keepdim=True).values
        mfcc_max = mfcc_tensor.max(dim=0, keepdim=True).values
        mfcc_normalized = (mfcc_tensor - mfcc_min) / (mfcc_max - mfcc_min)

        self.mfcc_features = mfcc_normalized
        return mfcc_normalized
    
    def extract_melspectrogram(self, frames, batch_size=60000, normalize=True):
        num_frames = frames.shape[0]
        n_mels = self.melspectrogram_transform.n_mels  

        mel_tensor = torch.zeros((num_frames, n_mels), device=self.device)

        for i in range(0, num_frames, batch_size):
            batch_frames = frames[i:i + batch_size].to(self.device)  
            mel_batch = self.melspectrogram_transform(batch_frames).mean(dim=-1)  
            mel_tensor[i:i + mel_batch.shape[0]] = mel_batch  

        if not normalize:
            self.melspectrogram_features = mel_tensor
            return mel_tensor

        mel_min = mel_tensor.min(dim=0, keepdim=True).values
        mel_max = mel_tensor.max(dim=0, keepdim=True).values
        mel_normalized = (mel_tensor - mel_min) / (mel_max - mel_min + 1e-6)

        self.melspectrogram_features = mel_normalized
        return mel_normalized

    def plot_mfccs(self):   
        mfcc_data = self.mfcc_features.cpu().numpy()

        plt.figure(figsize=(12, 8))
        plt.imshow(mfcc_data.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Frames')
        plt.ylabel('MFCCs')
        plt.title('MFCC Heatmap')
        plt.show()

    def plot_melspectrogram(self):
        mel_data = self.melspectrogram_features.cpu().numpy()

        plt.figure(figsize=(12, 8))
        plt.imshow(mel_data.T, aspect='auto', origin='lower', cmap='viridis')  
        plt.colorbar(label='Amplitude')
        plt.xlabel('Frames')
        plt.ylabel('Mel Frequency')
        plt.title('Mel Spectrogram Heatmap')
        plt.show()


