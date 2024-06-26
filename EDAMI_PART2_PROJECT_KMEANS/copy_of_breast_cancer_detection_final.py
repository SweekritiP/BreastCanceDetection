# -*- coding: utf-8 -*-
"""Copy of Breast_cancer_detection_Final

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bkXMGoM6_nKfAy_90z6qnFSvHt7X4wCx

Importing necesarry libarary
"""

!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo, list_available_datasets
import ucimlrepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

list_available_datasets(search='Bre')

cancer = fetch_ucirepo(id=15)
cancer.data.original

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)

#Shape of the Dataset
df.shape

"""Data pre-processing"""

df.drop(['id'],axis=1,inplace = True)

df.columns

df.info()

#Diagnosis class Malignant = 4 and Benign = 2
#The number of Benign and Maglinant cases from the dataset
df['class'].value_counts()

df['bare_nuclei'].value_counts()

df[df['bare_nuclei'] == '?']

df[df['bare_nuclei'] == '?'].sum()

df.replace('?',np.nan,inplace=True)

df['bare_nuclei'][23]

df.isna().sum()

df.fillna(method='ffill', inplace=True)

df.isna().sum()

df['bare_nuclei'] = df['bare_nuclei'].astype('int64')

df.info()

df.describe()

"""Bivariate Data Analysis"""

import seaborn as sns

sns.displot(df['class'],kde=True)

ax = df[df['class'] == 4][0:50].plot(kind='scatter', x='clump_thickness', y='uniform_cell_size', color='DarkBlue', label='malignant');
df[df['class'] == 2][0:50].plot(kind='scatter', x='clump_thickness', y='uniform_cell_size', color='Yellow', label='benign', ax=ax);
plt.show()

"""Multivariate Data Analysis"""

# Plot histograms for each variable
sns.set_style('darkgrid')
df.hist(figsize=(30,30))
plt.show()

"""from pandas.plotting import scatter_matrix"""

from pandas.plotting import scatter_matrix
# Create scatter plot matrix
scatter_matrix(df, figsize = (18,18))
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(data=df,orient='h')

df.corr()

plt.figure(figsize=(30,20))
cor = df.corr()
sns.heatmap(cor,vmax=1,square = True,annot=True, cmap=plt.cm.Blues)
plt.title('Correlation between different attributes')
plt.show()

#Correlation with output variable
cor_target = abs(cor["class"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0]
relevant_features

df.shape

df.columns

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class CustomKMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            self.labels = self._assign_clusters(X)

            # Update centroids based on the mean of data points in each cluster
            new_centroids = self._update_centroids(X, self.labels)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self.labels

    def _initialize_centroids(self, X):
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def _assign_clusters(self, X):
        distances = np.sqrt(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2))
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)

        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = X[np.random.choice(len(X))]

        return new_centroids

    def compute_wcss(self, X, labels):
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - self.centroids[i]) ** 2)
        return wcss

    def predict(self, X):
        X = np.array(X)  # Convert input to NumPy array
        return self._assign_clusters(X)

# Assuming df is your DataFrame containing the data
features = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses']
new_df = df[features]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(new_df.values)

# Perform dynamic k-means clustering for 2 clusters
n_clusters = 2
custom_kmeans = CustomKMeans(n_clusters=n_clusters)
labels = custom_kmeans.fit(X)

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Save the scaler and the model using pickle
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(custom_kmeans, model_file)

# Function to predict using the CustomKMeans model
def predict_tumor_type(new_data):
    # Load the scaler and model
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open('model.pkl', 'rb') as model_file:
        custom_kmeans = pickle.load(model_file)

    new_data_scaled = scaler.transform(new_data)
    clusters = custom_kmeans.predict(new_data_scaled)
    return clusters

# Example usage:
# New data should be a 2D array where each row is a new sample
new_data = [[5,4,1,4,5,4,4,2,3]]  # Replace with actual new data
predictions = predict_tumor_type(new_data)
for i, prediction in enumerate(predictions):
    print(f'Sample {i+1}: Cluster {prediction}')



import pickle
import numpy as np

# Load the scaler and model from the file
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict using the model directly
def predict_with_model(input_data):
    # Ensure the input data is a NumPy array
    input_data = np.array(input_data)

    # Scale the input data using the same scaler
    scaled_data = scaler.transform(input_data)

    # Predict using the model
    return model.predict(scaled_data)

# Example usage
new_data = [[5, 4, 6, 4, 5, 33, 4, 2, 3], [2, 5, 2, 4, 1, 2, 2, 4, 5]]
predictions = predict_with_model(new_data)
for i, prediction in enumerate(predictions):
    print(f'Sample {i+1}: Cluster {prediction}')

df
df.shape

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Extract 'Class' column from dx and 'Cluster' column from d
class_column1 = df['class']
cluster_column1 = df['Cluster']

# Create a new DataFrame
new_df1=pd.DataFrame({'class':class_column1,'Cluster':cluster_column1})
new_df1
new_df1['class_mapped'] = new_df1['class'].map({4: 1, 2: 0})
new_df1

new_df1.iloc[5:20]

new_df1.groupby('class').size()

from sklearn.metrics import confusion_matrix, accuracy_score

# Extract 'Class' and 'Cluster' columns
true_labels = new_df1['class_mapped']
predicted_labels = new_df1['Cluster']

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate accuracy score
accuracy = accuracy_score(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy Score:", accuracy)
# Create confusion matrix
conf_matrix = confusion_matrix(new_df1['class_mapped'], new_df1['Cluster'])

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Class vs Cluster')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

new_df1.iloc[0:20]