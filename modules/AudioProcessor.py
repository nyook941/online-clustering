import torchaudio
import torch
import numpy as np

class AudioProcessor:
    def __init__(self, audio_files, frame_size_ms=150, hop_length_ratio=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_files = audio_files

        self.frame_size_ms = frame_size_ms
        self.hop_length_ratio = hop_length_ratio

        self.waveforms = []
        self.sample_rate = 48000
        
        self.frames = None
        self.class_ids = None
        self.timestamps = None

        self.load_audios()

    def load_audios(self):
        for i, file in enumerate(self.audio_files):
            waveform, self.sample_rate = torchaudio.load(file)
            waveform.to(self.device)
            waveform = torch.mean(waveform, dim=0)
            self.waveforms.append(waveform)
            self.create_frames(waveform, class_id=i)

    def create_frames(self, waveform, class_id):
        frame_size = int((self.frame_size_ms * self.sample_rate) / 1000)
        hop_length = int((self.frame_size_ms * (1-self.hop_length_ratio) * self.sample_rate) / 1000)
        
        num_frames = (waveform.shape[0] - frame_size) // hop_length + 1
        frames = torch.zeros(num_frames, frame_size).to(self.device)
        timestamps = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_length
            timestamp = start / self.sample_rate
            frames[i, :] = waveform[start:start + frame_size]
            timestamps[i] = timestamp

        class_ids = np.full((num_frames,), class_id)

        if self.frames is None:
            self.frames = frames
            self.class_ids = class_ids
            self.timestamps = timestamps
        else:
            self.frames = torch.vstack((self.frames, frames))
            self.class_ids = np.hstack((self.class_ids, class_ids))
            self.timestamps = np.hstack((self.timestamps, timestamps))
