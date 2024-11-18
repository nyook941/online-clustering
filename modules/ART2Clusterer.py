import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
import ast
import numpy as np

class ART2Clusterer:
    def __init__(self, vigilance, n_features, time_series_index, max_clusters = 100, predicted_filename_prefix="time_series"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predicted_filename = f"output/no_pca/{predicted_filename_prefix}_{time_series_index}/predicted_{time_series_index}.csv"
        self.df = pd.DataFrame(columns=['timestamp', 'predicted', 'ground_truth', 'features'])
        self.df.to_csv(self.predicted_filename)

        self.vigilance = vigilance
        
        self.cluster_means = torch.zeros(max_clusters, n_features).to(self.device)
        self.cluster_sizes = []
        self.class_mappings = {}

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
            self.initialize_first_cluster(class_id, timestamp, features)
            return
        distance, cluster_index = self.winner_mean(features)
        if self.vigilance_test(distance):
            self.add_to_cluster(timestamp, cluster_index, class_id, features)
        else:
            self.create_new_cluster(timestamp, class_id, features)
        
    def initialize_first_cluster(self, ground_truth_id, timestamp, features):
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

        rows = row_ind[:3]
        cols = col_ind[:3]
        conf_matrix = conf_matrix[np.ix_(rows, cols)]
        
        correctly_classified = conf_matrix.trace()
        total_samples = len(self.df)
        accuracy = correctly_classified / total_samples

        for i in range(3):
            self.class_mappings[cols[i]] = rows[i]

        return conf_matrix, accuracy
    
    def plot_truth_vs_time(self, window_title):
        true_labels = self.df['ground_truth'].values
        cluster_labels = self.df['predicted'].values
        cluster_labels = np.vectorize(lambda x: self.class_mappings.get(x, -1))(cluster_labels)
        times = self.df['timestamp'].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.canvas.manager.set_window_title(window_title)
        
        ax1.scatter(times, true_labels, color="blue", alpha=0.6, s=1)
        ax1.set_title("Ground Truth Labels")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Class ID")
        
        ax2.scatter(times, cluster_labels, color="red", alpha=0.6, s=1)
        ax2.set_title("Predicted Labels")
        ax2.set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.show()

class ART2ClustererSmoothing(ART2Clusterer):
    def __init__(self, vigilance, n_features, time_series_index, max_clusters=100, predicted_filename_prefix="time_series", buffer_size=5):
        super().__init__(vigilance, n_features, time_series_index, max_clusters, predicted_filename_prefix)

        # The first two indicies will be used to store class_id and timestamp, the rest of the indicies are features
        self.buffer = torch.zeros(buffer_size, n_features+2).to(self.device)

        self.buffer_mean = torch.zeros(n_features).to(self.device)
        self.buffer_size = buffer_size
        self.buffer_count = 0

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

        self.add_to_buffer(class_id, timestamp, features)

        if self.buffer_count < self.buffer_size:
            return
        
        if not self.cluster_sizes:
            self.initialize_first_cluster()
            self.reset_buffer()
            return
        distance, predicted = self.winner_mean(self.buffer_mean)
        if self.vigilance_test(distance):
            self.add_to_cluster(predicted)
        else:
            self.create_new_cluster()
        self.reset_buffer()
    
    def initialize_first_cluster(self):
        self.cluster_means[0,:] = self.buffer_mean
        self.cluster_sizes.append(self.buffer_count)

        self.add_to_csv(0)

    def add_to_cluster(self, predicted):
        current_size = self.cluster_sizes[predicted]
        current_mean = self.cluster_means[predicted]
        new_mean = (current_size * current_mean + self.buffer_size * self.buffer_mean) / (current_size + self.buffer_size)

        self.cluster_means[predicted,:] = new_mean
        self.cluster_sizes[predicted] += 1
        self.add_to_csv(predicted)

    def create_new_cluster(self):
        new_cluster_i = len(self.cluster_sizes)

        self.cluster_sizes.append(self.buffer_size)
        self.cluster_means[new_cluster_i, :] = self.buffer_mean
        self.add_to_csv(new_cluster_i)

    def add_to_csv(self, predicted):
        ground_truth = self.buffer[:,0].tolist()
        timestamp = self.buffer[:,1].tolist()


        for i in range(len(ground_truth)):
            features = self.buffer[i,2:].squeeze().tolist()
            data = {
                'timestamp': timestamp[i],
                'predicted': predicted,
                'ground_truth': ground_truth[i],
                'features': str(features)
            }
            self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)

    def add_to_buffer(self, class_id, timestamp, features):
        self.buffer[self.buffer_count, 0] = class_id
        self.buffer[self.buffer_count, 1] = timestamp
        self.buffer[self.buffer_count, 2:] = features

        self.buffer_mean = self.buffer[:, 2:].mean(dim=0)
        self.buffer_count += 1

    def reset_buffer(self):
        self.buffer.zero_()
        self.buffer_mean.zero_()
        self.buffer_count = 0