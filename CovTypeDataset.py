from __future__ import print_function, division
import copy
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CovTypeDataset(Dataset):
    """The Forest CoverType dataset"""

    def __init__(self, csv_file, device='cuda'):
        """
        Args:
            csv_file (string): Path to the csv file *with headers*.
        """
        
        # Import dataset and apply preprocessing
        self.data = pd.read_csv(csv_file, index_col=False)

        self.data['Cover_Type'] = self.data['Cover_Type'] - 1 # Set targets between 0-6
        self.processedData = self.data

        self.labels = torch.tensor(self.data['Cover_Type'], device=device)

        # Standardise data and move to GPU
        scaler = StandardScaler()
        scaled_continuous_vars = pd.DataFrame(scaler.fit_transform(self.data.iloc[:, :10].values.astype(np.float32), self.data['Cover_Type']), columns=self.data.columns[:10])
        self.data = pd.concat([scaled_continuous_vars, self.data.iloc[:, 10:-1]], axis=1) # Recombine standardised continuous variables with one hot encoded vars
        self.data = torch.tensor(self.data.values.astype(np.float32), device=device)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def random_split(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=1)
        train_set = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)
        return train_set, test_set

    def random_split_2_sets(self, test_size):
        X_set1, X_set2, y_set1, y_set2 = train_test_split(self.data, self.labels, test_size=0.5, random_state=1)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_set1, y_set1, test_size=test_size, random_state=1)
        train_set1 = TensorDataset(X_train1, y_train1)
        test_set1 = TensorDataset(X_test1, y_test1)

        X_train2, X_test2, y_train2, y_test2 = train_test_split(X_set2, y_set2, test_size=test_size, random_state=1)
        train_set2 = TensorDataset(X_train2, y_train2)
        test_set2 = TensorDataset(X_test2, y_test2)
        
        return train_set1, test_set1, train_set2, test_set2

class PermutedCovTypeDataset(CovTypeDataset):

    def __init__(self, csv_file, permute_columns_count, device='cuda'):
        """
        Args:
            csv_file (string): Path to the csv file *with headers*.
            permute_columns_count (int): Number of columns to permute.
        """
        super().__init__(csv_file, device)
        permuted_data = pd.DataFrame()
        for n in range(len(self.processedData.columns) -1):
            data_slice = copy.deepcopy(self.processedData.iloc[:, n])
            if n <= permute_columns_count:
                data_slice.sort_values(inplace=True, ignore_index=True)
            permuted_data = pd.concat([permuted_data, data_slice], axis=1)
        data_slice = copy.deepcopy(self.processedData.iloc[:, -1])
        permuted_data = pd.concat([permuted_data, data_slice], axis=1)

        self.labels = torch.tensor(permuted_data.iloc[:, -1], device=device)

        # Standardise data and move to GPU
        scaler = StandardScaler()
        scaled_continuous_vars = pd.DataFrame(scaler.fit_transform(permuted_data.iloc[:, :10].values.astype(np.float32), permuted_data.iloc[:, -1]))
        self.data = pd.concat([scaled_continuous_vars, permuted_data.iloc[:, 10:-1]], axis=1) # Recombine standardised continuous variables with one hot encoded vars
        self.data = torch.tensor(self.data.values.astype(np.float32), device=device)