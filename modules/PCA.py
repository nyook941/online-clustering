import torch
import pandas as pd
import ast

class PCA:
    def __init__(self, csv_filename, variance=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df = pd.read_csv(csv_filename)
        self.features = self.get_features()
        self.standardized_features = self.standardize()
        self.covariance_matrix = self.covariance_matrix()

        self.eig_vals, self.eig_vectors = self.eigen()

        self.principle_components = self.get_principle_components(variance)
        self.projected_data = self.project_data()

    def get_features(self):
        mfccs = self.df["mfccs"].apply(lambda x: ast.literal_eval(x)).tolist()
        mfccs = torch.tensor(mfccs).to(self.device)
        
        mfsc = self.df["mfsc"].apply(lambda x: ast.literal_eval(x)).tolist()
        mfsc = torch.tensor(mfsc).to(self.device)

        return torch.cat((mfccs, mfsc), dim=1)
    
    def standardize(self):
        means = self.features.mean(dim=0)
        std = self.features.std(dim=0)
        return (self.features - means) / std
    
    def covariance_matrix(self):
        return (self.standardized_features.T @ self.standardized_features) / (self.standardized_features.shape[0] - 1)

    def eigen(self):
        eig_vals, eig_vecs = torch.linalg.eig(self.covariance_matrix)
        eig_vals = eig_vals.real
        eig_vecs = eig_vecs.real
        sorted_indices = torch.argsort(eig_vals, descending=True)
        return eig_vals[sorted_indices], eig_vecs[:, sorted_indices]
    
    def get_principle_components(self, variance=0.95):
        explained_variance = self.eig_vals / torch.sum(self.eig_vals)
        cum_explained_variance = torch.cumsum(explained_variance, dim=0)

        k = torch.nonzero(cum_explained_variance >= variance)[0].item() + 1
        return self.eig_vectors[:, :k]
    
    def project_data(self):
        return torch.matmul(self.standardized_features, self.principle_components)