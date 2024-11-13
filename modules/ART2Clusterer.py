import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import ast

class ART2Clusterer:
    def __init__(self, vigilance, n_features, time_series_index, max_clusters = 100, predicted_filename_prefix="time_series"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predicted_filename = f"output/{predicted_filename_prefix}_{time_series_index}/predicted_{time_series_index}.csv"
        self.df = pd.DataFrame(columns=['timestamp', 'predicted', 'ground_truth', 'features'])
        self.df.to_csv(self.predicted_filename)

        self.vigilance = vigilance
        
        self.cluster_means = torch.zeros(max_clusters, n_features).to(self.device)
        self.cluster_sizes = []

    def add_to_csv(self, timestamp, predicted, ground_truth_id, features):
        features_list = features.squeeze().tolist()
        data = {
            'timestamp': timestamp,
            'predicted': predicted,
            'ground_truth': ground_truth_id,
            'features': str(features_list)
        }

        self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)

    def write_to_csv(self):
        self.df.to_csv(self.predicted_filename)

    def fit_clusters(self, datapoint):
        class_id = datapoint['class_id']
        timestamp = datapoint['timestamp']

        features = None
        if 'mfccs' in datapoint:
            mfccs = ast.literal_eval(datapoint['mfccs'])
            mfsc = ast.literal_eval(datapoint['mfsc'])
            features = torch.tensor(mfccs + mfsc).to(self.device)
        else:
            features = ast.literal_eval(datapoint['features'])
            features =  torch.tensor(features).to(self.device)

        if not self.cluster_sizes:
            self._initialize_first_cluster(class_id, timestamp, features)
            return
        distance, cluster_index = self.winner_mean(features)
        if self.vigilance_test(distance):
            self.add_to_cluster(timestamp, cluster_index, class_id, features)
        else:
            self.create_new_cluster(timestamp, class_id, features)
        
    def _initialize_first_cluster(self, ground_truth_id, timestamp, features):
        self.cluster_sizes.append(1)
        self.cluster_means[0,:] = features

        self.add_to_csv(timestamp, 0, ground_truth_id, features)

    def winner_mean(self, x):
        valid_means = self.cluster_means[0:len(self.cluster_sizes)]
        euclidean_distances = torch.sqrt(torch.sum(torch.pow(valid_means - x, 2), dim=1))
        min_distance = torch.min(euclidean_distances).item()
        min_index = torch.argmin(euclidean_distances).item()
        return min_distance, min_index
    
    def vigilance_test(self, distance_winner):
        return distance_winner ** 2 < self.vigilance
    
    def create_new_cluster(self, timestamp, class_id, x):
        # get the first unused cluster
        new_cluster_i = len(self.cluster_sizes)

        # assign the mean and actual data to the cluster
        self.cluster_sizes.append(1)
        self.cluster_means[new_cluster_i,:] = x
        self.add_to_csv(timestamp, new_cluster_i, class_id, x)

    def add_to_cluster(self, timestamp, cluster_index, class_id, x):
        # cacluate the new mean
        current_size = self.cluster_sizes[cluster_index]
        current_mean = self.cluster_means[cluster_index]
        new_mean = (current_mean * current_size + x) / (current_size + 1)
        
        # assign the data to the cluster
        self.cluster_means[cluster_index,:] = new_mean
        self.cluster_sizes[cluster_index] += 1
        self.add_to_csv(timestamp, cluster_index, class_id, x)      

    def calc_accuracy(self):
        true_labels = self.df['ground_truth'].tolist()
        cluster_labels = self.df['predicted'].tolist()

        conf_matrix = confusion_matrix(true_labels, cluster_labels)
        
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        mapping = {col: row for row, col in zip(row_ind, col_ind)}

        mapped_labels = [mapping[cluster] for cluster in cluster_labels]
        accuracy = accuracy_score(true_labels, mapped_labels)
        
        return conf_matrix, accuracy
    
    def plot_truth_vs_time(self):
        true_labels = self.df['ground_truth'].values
        cluster_labels = self.df['predicted'].values
        times = self.df['timestamp'].values

        times = true_labels * 120 + times

        times = np.sort(times)
        true_labels = np.sort(true_labels)
        cluster_labels = np.sort(cluster_labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        ax1.scatter(times, true_labels, color="blue", alpha=0.6, s=1)
        ax1.set_title("Ground Truth Labels")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Class ID")
        
        ax2.scatter(times, cluster_labels, color="red", alpha=0.6, s=1)
        ax2.set_title("Predicted Labels")
        ax2.set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.show()
    
class SmoothingART2Clusterer(ART2Clusterer):
    def __init__(self, vigilance, n_features, n_clusters=10, max_samples_per_cluster=4000, buffer_size=5):
        super().__init__(vigilance, n_features, n_clusters, max_samples_per_cluster)
        self.buffer = torch.zeros(buffer_size, n_features).to(self.device)
        self.buffer_indicies = torch.zeros(buffer_size).to(self.device)

        self.buffer_mean = torch.zeros(n_features).to(self.device)
        self.buffer_size = buffer_size
        self.buffer_count = 0

    def fit_clusters(self, x, i):
        x = x.to(self.device)
        self.add_to_buffer(x, i)
        if self.buffer_count < self.buffer_size:
            return

        if self.cluster_sizes[0].item() == 0:
            self._initialize_first_cluster()
        else:
            winner, distance, cluster_index = self.winner_mean(self.buffer_mean)
            if self.vigilance_test(distance):
                self.add_to_cluster(cluster_index)
            else:
                self.create_new_cluster()

        self.reset_buffer()

    def _initialize_first_cluster(self):
        self.clusters[0, 0, :] = self.buffer_mean

        self.clusters[0, 1:self.buffer_count+1, :] = self.buffer

        self.cluster_sizes[0] = self.buffer_count
        self.cluster_data_indicies[0, 1:self.buffer_count+1] = self.buffer_indicies

    def add_to_cluster(self, cluster_i):
        # cacluate the new mean
        current_size = self.cluster_sizes[cluster_i].item()
        current_mean = self.clusters[cluster_i, 0]
        new_mean = (current_mean * current_size + self.buffer_mean * self.buffer_count) / (current_size + self.buffer_count)
        self.clusters[cluster_i, 0] = new_mean

        # assign the index of the data
        self.cluster_data_indicies[cluster_i, current_size+1:current_size+self.buffer_size+1] = self.buffer_indicies
        
        # assign the data to the cluster
        self.clusters[cluster_i, current_size+1 : current_size+self.buffer_size+1] = self.buffer
        self.cluster_sizes[cluster_i] += self.buffer_count

    def create_new_cluster(self):
        # get the first unused cluster
        new_cluster_i = torch.where(self.cluster_sizes == 0)[0] 
        new_cluster_i = new_cluster_i[0]

        # assign the index associated with the data point
        self.cluster_data_indicies[new_cluster_i, 1:self.buffer_count+1] = self.buffer_indicies

        # assign the mean and actual data to the cluster
        self.clusters[new_cluster_i, 0] = self.buffer_mean
        self.clusters[new_cluster_i, 1:self.buffer_count+1] = self.buffer
        self.cluster_sizes[new_cluster_i] = self.buffer_count

    def add_to_buffer(self, x, i):
        self.buffer[self.buffer_count,:] = x
        self.buffer_indicies[self.buffer_count] = i

        self.buffer_mean = (self.buffer_mean * self.buffer_count + x)/(self.buffer_count+1) 
        self.buffer_count += 1

    def reset_buffer(self):
        self.buffer.zero_()
        self.buffer_indicies.zero_()
        self.buffer_mean.zero_()
        self.buffer_count = 0