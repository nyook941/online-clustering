import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioProcessor:
    def __init__(self, audio_files, frame_size_ms=150, hop_length_ratio=0.1):
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
            waveform, self.sample_rate = librosa.load(file, sr=None)
            self.waveforms.append(waveform)
            self.create_frames(waveform, class_id=i)

    def create_frames(self, waveform, class_id):
        frame_size = int((self.frame_size_ms * self.sample_rate) / 1000)
        hop_length = int((self.frame_size_ms * self.hop_length_ratio * self.sample_rate) / 1000)
        
        stft_result = librosa.stft(waveform, n_fft=frame_size, hop_length=hop_length)
        magnitude, _ = librosa.magphase(stft_result)

        num_frames = magnitude.shape[1]
        frame_data = []
        timestamps = []

        for i in range(num_frames):
            timestamp = i * hop_length / self.sample_rate
            timestamps.append(timestamp)

            frame_data.append(magnitude[:, i])

        class_ids = np.full((num_frames,), class_id)
        if self.frames is None:
            self.frames = np.array(frame_data)
            self.class_ids = class_ids
            self.timestamps = np.array(timestamps)
        else:
            self.frames = np.vstack((self.frames, frame_data))
            self.class_ids = np.hstack((self.class_ids, class_ids))
            self.timestamps = np.hstack((self.timestamps, timestamps))

    def shuffle_frames(self):
        indices = np.arange(len(self.frames))
        np.random.shuffle(indices)
        
        self.frames = self.frames[indices]
        self.class_ids = self.class_ids[indices]
        self.timestamps = self.timestamps[indices]
        
        return self.frames, self.class_ids, self.timestamps

    def plot_waveform(self):
        if len(self.waveforms) > 0:
            plt.figure(figsize=(10, 4))
            for i, waveform in enumerate(self.waveforms):
                plt.plot(waveform, label=f'Audio {i}')
            plt.title('Waveform')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()
        else:
            print("Waveform is not loaded.")