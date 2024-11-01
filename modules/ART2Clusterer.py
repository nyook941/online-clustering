import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np

class ART2Clusterer:
    def __init__(self, vigilance, n_features, n_clusters=10, max_samples_per_cluster=4000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vigilance = vigilance
        self.n_clusters = n_clusters

        self.clusters = torch.zeros(n_clusters, max_samples_per_cluster, n_features).to(self.device)
        self.cluster_data_indicies = torch.full((n_clusters, max_samples_per_cluster), -1).to(self.device)
        self.cluster_sizes = torch.zeros(n_clusters, dtype=torch.long).to(self.device)
        
    def _initialize_first_cluster(self, initial_data, i):
        self.clusters[0, 0, :] = initial_data.to(self.device) # place the mean in the first index
        self.clusters[0, 1, :] = initial_data.to(self.device) # place the actual data in the cluster
        
        self.cluster_sizes[0] = 1

        self.cluster_data_indicies[0, 1] = i

    def winner_mean(self, x):
        valid_means = self.clusters[self.cluster_sizes > 0, 0]
        euclidean_distances = torch.sqrt(torch.sum(torch.pow(valid_means - x, 2), dim=1))
        min_index = torch.argmin(euclidean_distances)
        return valid_means[min_index], euclidean_distances[min_index], min_index
    
    def vigilance_test(self, distance_winner):
        return torch.pow(distance_winner, 2) < self.vigilance
    
    def create_new_cluster(self, x, i):
        # get the first unused cluster
        new_cluster_i = torch.where(self.cluster_sizes == 0)[0] 
        new_cluster_i = new_cluster_i[0] 

        # assign the index associated with the data point
        self.cluster_data_indicies[new_cluster_i, 1] = i

        # assign the mean and actual data to the cluster
        self.clusters[new_cluster_i, 0] = x.to(self.device)
        self.clusters[new_cluster_i, 1] = x.to(self.device)
        self.cluster_sizes[new_cluster_i] = 1

    def add_to_cluster(self, x, cluster_i, i):
        # cacluate the new mean
        current_size = self.cluster_sizes[cluster_i].item()
        current_mean = self.clusters[cluster_i, 0]
        new_mean = (current_mean * current_size + x) / (current_size + 1)
        self.clusters[cluster_i, 0] = new_mean

        # assign the index of the data
        self.cluster_data_indicies[cluster_i, current_size+1] = i
        
        # assign the data to the cluster
        self.clusters[cluster_i, current_size + 1, :] = x.to(self.device)
        self.cluster_sizes[cluster_i] += 1

    def fit_clusters(self, x, i):
        x = x.to(self.device)
        if self.cluster_sizes[0].item() == 0:
            self._initialize_first_cluster(x, i)
            return
        winner, distance, cluster_index = self.winner_mean(x)
        if self.vigilance_test(distance):
            self.add_to_cluster(x, cluster_index, i)
        else:
            self.create_new_cluster(x, i)

    def calc_accuracy(self, ground_truth_csv):
        true_labels, cluster_labels, _ = self._get_labels_and_times(ground_truth_csv)

        conf_matrix = confusion_matrix(true_labels, cluster_labels)
        
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        mapping = {col: row for row, col in zip(row_ind, col_ind)}

        mapped_labels = [mapping[cluster] for cluster in cluster_labels]
        accuracy = accuracy_score(true_labels, mapped_labels)
        
        return conf_matrix, accuracy
    
    def plot_truth_vs_time(self, ground_truth_csv):
        y_true, y_pred, times = self._get_labels_and_times(ground_truth_csv)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        ax1.scatter(times, y_true, color="blue", alpha=0.6, s=1)
        ax1.set_title("Ground Truth Labels")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Class ID")
        
        ax2.scatter(times, y_pred, color="red", alpha=0.6, s=1)
        ax2.set_title("Predicted Labels")
        ax2.set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.show()

    def _get_labels_and_times(self, ground_truth_csv):
        df = pd.read_csv(ground_truth_csv)
        ground_truth_indices = df.iloc[:, 0].values
        ground_truth_labels = df['class_id'].values
        timestamps = df['timestamp'].values

        actual_times = ground_truth_labels * 120 + timestamps

        cluster_sizes = self.cluster_sizes.cpu().numpy()
        cluster_data_indices = [index.cpu().numpy() for index in self.cluster_data_indicies]

        valid_clusters = [i for i, size in enumerate(cluster_sizes) if size > 0]

        cluster_labels = []
        true_labels = []

        for cluster_i in valid_clusters:
            indices = cluster_data_indices[cluster_i][1:cluster_sizes[cluster_i]+1]  # Skip the mean
            for i in indices:
                true_class = ground_truth_labels[ground_truth_indices == i].item()
                true_labels.append(true_class)
                cluster_labels.append(cluster_i)

        return np.sort(true_labels), np.sort(cluster_labels), np.sort(actual_times)
    
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