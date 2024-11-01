import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
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

    def calc_confusion_matrix(self, ground_truth_csv):
        df = pd.read_csv(ground_truth_csv)
        ground_truth_indices = df.iloc[:, 0].values
        ground_truth_labels = df['class_id'].values

        y_true = []
        y_pred = []

        for i in range(self.n_clusters):
            cluster_size = self.cluster_sizes[i].item()
            if cluster_size > 0:
                data_i = self.cluster_data_indicies[i, 1:cluster_size+1].cpu().numpy()

                ground_truth = ground_truth_labels[np.isin(ground_truth_indices, data_i)]

                if len(ground_truth) > 0:
                    most_common_label = mode(ground_truth)[0]

                    y_true.extend(ground_truth)
                    y_pred.extend([most_common_label] * len(ground_truth))
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        return conf_matrix     

    def calc_accuracy(self, conf_matrix):
        correct_predictions = np.trace(conf_matrix)
        total_predictions = np.sum(conf_matrix)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy
    
    def plot_truth_vs_time(self, ground_truth_csv):
        # Use the helper function to get sorted y_true, y_pred, and times
        y_true, y_pred, times = self._get_labels_and_times(ground_truth_csv)

        # Create side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        # Plot ground truth on the first subplot
        ax1.scatter(times, y_true, color="blue", alpha=0.6, s=1)
        ax1.set_title("Ground Truth Labels")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Class ID")
        
        # Plot predicted labels on the second subplot
        ax2.scatter(times, y_pred, color="red", alpha=0.6, s=1)
        ax2.set_title("Predicted Labels")
        ax2.set_xlabel("Time (seconds)")

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def _get_labels_and_times(self, ground_truth_csv):
        # Load ground truth data
        df = pd.read_csv(ground_truth_csv)
        indices = df.iloc[:, 0].values  # First unnamed column as indices
        class_ids = df['class_id'].values  # Ground truth labels
        timestamps = df['timestamp'].values  # Timestamps in seconds

        # Calculate the adjusted time with the formula `actual_time = class_id * 120 + timestamp`
        actual_times = class_ids * 120 + timestamps

        y_true = []
        y_pred = []
        unique_times = []

        # Map each cluster to the most common ground truth label
        for cluster_idx in range(self.n_clusters):
            cluster_size = self.cluster_sizes[cluster_idx].item()
            if cluster_size > 0:
                data_indices = self.cluster_data_indicies[cluster_idx, 1:cluster_size+1].cpu().numpy()
                
                # Get ground truth labels and timestamps for points in this cluster
                cluster_ground_truth = class_ids[np.isin(indices, data_indices)]
                cluster_timestamps = actual_times[np.isin(indices, data_indices)]

                if len(cluster_ground_truth) > 0:
                    # Determine the most common ground truth label for this cluster
                    most_common_label = mode(cluster_ground_truth)[0]
                    
                    # Extend the lists for ground truth, predicted labels, and timestamps
                    y_true.extend(cluster_ground_truth)
                    y_pred.extend([most_common_label] * len(cluster_ground_truth))
                    unique_times.extend(cluster_timestamps)

        # Sort by time for a smoother plot
        sorted_indices = np.argsort(unique_times)
        sorted_times = np.array(unique_times)[sorted_indices]
        sorted_y_true = np.array(y_true)[sorted_indices]
        sorted_y_pred = np.array(y_pred)[sorted_indices]

        return sorted_y_true, sorted_y_pred, sorted_times
    
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