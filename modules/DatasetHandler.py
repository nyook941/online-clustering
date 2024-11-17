import os
import shutil
import torch
import pandas as pd
import random

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

    def generate_all_features_csv(self, filename="all_features.csv"):
        mfccs_cpu = self.mfcc_features.cpu().numpy().tolist()
        mel_cpu = self.mel_features.cpu().numpy().tolist()
        
        data = {
            "class_id": self.class_ids,
            "timestamp": self.timestamps,
            "mfccs": mfccs_cpu,
            "mfsc": mel_cpu
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_dir + "/" + filename, index=False)

    def generate_pca_csv(self, features, filename="pca_features.csv"):
        features = features.cpu().numpy().tolist()
        data = {
            "class_id": self.class_ids,
            "timestamp": self.timestamps,
            "features": features,
        }

        df = pd.DataFrame(data)
        df.to_csv(self.output_dir + "/" + filename, index=False)

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
        df.to_csv(self.output_dir + "/" + filename, index=False)

    def generate_time_series(self, n_time_series=5, time_series_duration=120, filename="selected_features.csv", output_folder = "time_series"):
        for i in range(n_time_series):
            duration_sum = 0
            remaining_classes = [0, 1, 2]
            df = pd.read_csv(self.output_dir + "/" + filename)
            time_series_data = pd.DataFrame(columns=df.columns)

            while duration_sum < time_series_duration:
                # choose class
                if not remaining_classes:
                    remaining_classes = [0, 1, 2]
                chosen_class = random.choice(remaining_classes)
                remaining_classes.remove(chosen_class)

                # choose time duration for that class
                time_duration = 15 if duration_sum + 15 > time_series_duration else random.randint(15, 30)
                class_data = df[df['class_id'] == chosen_class]
                selected_rows = class_data[(class_data['timestamp'] >= duration_sum & 
                                            (class_data['timestamp'] < duration_sum + time_duration))]
                
                if selected_rows.empty:
                    continue

                time_series_data = pd.concat([time_series_data, selected_rows], ignore_index=True)
                duration_sum += time_duration

            os.makedirs(self.output_dir + f"/{output_folder}_{i}")
            time_series_data.to_csv(self.output_dir + f'/{output_folder}_{i}/ground_truth_{i}.csv', index=False)

            

        
        # hop_length_ms = frame_size_ms * hop_ratio
        # num_frames = math.ceil(time_series_ms / hop_length_ms)

        # for i in range(n_time_series):
        #     df = self.shuffle_data(filename)
        #     selected_df = df.iloc[:num_frames]

        #     os.makedirs(self.output_dir + f"/{output_folder}_{i}")
        #     selected_df.to_csv(self.output_dir + f'/{output_folder}_{i}/ground_truth_{i}.csv', index=False)