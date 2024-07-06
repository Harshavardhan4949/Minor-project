Spotify Project

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer  # Import for imputation

# Check current working directory (optional)
import os
print("Current working directory:", os.getcwd())

# Load the dataset
spotify_data = pd.read_csv('spotify_dataset.csv')

# Check for missing values
missing_values = spotify_data.isnull().sum()
print("Missing values:\n", missing_values)

# Fill missing values in non-numeric columns with a placeholder
spotify_data.fillna({'track_name': 'Unknown'}, inplace=True)
spotify_data.fillna({'track_artist': 'Unknown'}, inplace=True)
spotify_data.fillna({'track_album_name': 'Unknown'}, inplace=True)
spotify_data.fillna({'track_album_id': 'Unknown'}, inplace=True)
spotify_data.fillna({'playlist_id': 'Unknown'}, inplace=True)

# **Improved Missing Value Handling:**
# 1. Use SimpleImputer to fill remaining missing values in numeric columns with the mean strategy
imputer = SimpleImputer(strategy='mean')
spotify_data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
              'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
              'duration_ms']] = imputer.fit_transform(spotify_data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                                                                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                                                                        'duration_ms']])

# 2. Log-transform skewed features (optional)
# If you identify skewed features after imputation, consider log-transformation to normalize the distribution
# spotify_data['column_name'] = np.log1p(spotify_data['column_name'])

# Convert relevant columns to numeric, ensuring any remaining missing values are handled
for col in spotify_data.columns:
  if spotify_data[col].dtype == 'object':
    try:
      # Attempt to convert to numeric
      spotify_data[col] = pd.to_numeric(spotify_data[col], errors='coerce')
    except ValueError:
      # If conversion fails, consider more advanced techniques (e.g., category encoding)
      # You can explore techniques like one-hot encoding or label encoding for categorical features
      # spotify_data[col] = pd.Categorical(spotify_data[col]).codes  # Example of label encoding

# Print the DataFrame before dropping NaNs
      print("Data before dropping NaNs:\n", spotify_data.head())

# **Adjusted Dropping Strategy:**
# 1. Handle outliers before dropping NaNs
# Clip outliers to a reasonable range to avoid issues with scaling
spotify_data = pd.DataFrame(np.clip(spotify_data, -100, 100), columns=spotify_data.columns)  # Clip outliers

# 2. Drop rows with a threshold of missing values (e.g., 20%)
spotify_data.dropna(thresh=0.8, inplace=True)  # Adjust threshold as needed

# Ensure DataFrame is not empty
print("Shape of spotify_data after cleaning:", spotify_data.shape)
if spotify_data.empty:
  raise ValueError("After cleaning, the DataFrame is empty. Check your data for issues.")

# Scale the numeric features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(spotify_data)

# Data Analysis and Visualizations
# Correlation Matrix
correlation_matrix = spotify_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# ... (rest of your code for clustering, model building, etc.)

# Based on the above analysis, choose the appropriate number of clusters
kmeans = KMeans(n_clusters=5, random_state=42)
spotify_data['cluster'] = kmeans.fit_predict(scaled_features)

# Plot clusters according to different parameters like playlist genres, playlist names
plt.scatter
