import torch
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, csv_filename, variance=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.df = pd.read_csv(csv_filename)
        self.features = self.get_features()
        self.standardized_features = self.standardize()
        
        self.pca_model = self.compute_pca(variance)
        self.projected_data = self.project_data()
    
    def get_features(self):
        mfccs = self.df["mfccs"].apply(lambda x: ast.literal_eval(x)).tolist()
        mfccs = np.array(mfccs)
        
        mfsc = self.df["mfsc"].apply(lambda x: ast.literal_eval(x)).tolist()
        mfsc = np.array(mfsc)
        
        features = np.hstack((mfccs, mfsc))
        return features
        
    def standardize(self):
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(self.features)
        return standardized_features

    def compute_pca(self, variance=0.95):        
        pca = SKPCA(n_components=variance)
        pca.fit(self.standardized_features)
        
        self.principal_components = pca.components_
        self.explained_variance_ratio = pca.explained_variance_ratio_
        return pca

    def project_data(self):
        projected_np = self.pca_model.transform(self.standardized_features)
        
        projected = torch.tensor(projected_np, dtype=torch.float32).to(self.device)
        
        projected_min = projected.min(dim=0, keepdim=True).values
        projected_max = projected.max(dim=0, keepdim=True).values
        normalized_projected = (projected - projected_min) / (projected_max - projected_min)
        
        return normalized_projected

    def plot_projected_data(self, graph_title):
        projected_data = self.projected_data.cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(projected_data.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Frames')
        plt.ylabel('Projected Data')
        plt.title(graph_title)
        plt.show()