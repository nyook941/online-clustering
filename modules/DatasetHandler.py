import os
import shutil
import torch
import math
import pandas as pd

class DatasetHandler:
    def __init__(self, class_ids, timestamps, mfcc_features, mel_features, n_features):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = "output"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)  
        os.makedirs(self.output_dir)  

        self.class_ids = class_ids
        self.timestamps = timestamps
        self.mfcc_features = mfcc_features
        self.mel_features = mel_features
        self.n_features = n_features

        self.time_series_tensor = None

    def generate_csv(self, filename="all_features.csv"):
        mfccs_cpu = self.mfcc_features.cpu().numpy().tolist()
        mel_cpu = self.mel_features.cpu().numpy().tolist()
        
        data = {
            "class_id": self.class_ids,
            "timestamp": self.timestamps,
            "mfccs": mfccs_cpu,
            "mfsc": mel_cpu
        }
        
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)

    def save_selected_features(self, mfcc_is, mfsc_is, filename="selected_features.csv"):
        selected_mfcc_features = self.mfcc_features[:, mfcc_is]
        selected_mel_features = self.mel_features[:, mfsc_is]  
        
        selected_mfcc_list = selected_mfcc_features.cpu().numpy().tolist()
        selected_mel_list = selected_mel_features.cpu().numpy().tolist()
        
        data = {
            "class_id": self.class_ids,
            "timestamp": self.timestamps,
            "mfccs": selected_mfcc_list,
            "mfsc": selected_mel_list
        }
        
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)

    def shuffle_data(self, filename):
        filepath = os.path.join(self.output_dir, filename)
        df = pd.read_csv(filepath)
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    def generate_time_series(self, frame_size_ms, hop_ratio, n_time_series=5, time_series_ms=120000, filename="selected_features.csv"):
        hop_length_ms = frame_size_ms * hop_ratio
        num_frames = math.ceil(time_series_ms / hop_length_ms)
        time_series_tensor = torch.zeros(n_time_series, num_frames, self.n_features).to(self.device)

        for i in range(n_time_series):
            df = self.shuffle_data(filename)
            selected_df = df.iloc[:num_frames]
            selected_df.to_csv(os.path.join(self.output_dir, f'ground_truth_{i}.csv'))

            mfccs = selected_df['mfccs'].apply(eval).tolist()
            mfsc = selected_df['mfsc'].apply(eval).tolist()

            features = torch.tensor([m + s for m, s in zip(mfccs, mfsc)], dtype=torch.float32)
            time_series_tensor[i, :features.shape[0], :] = features

        self.time_series_tensor = time_series_tensor
        return time_series_tensor
