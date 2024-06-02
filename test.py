import os
import umap
import numpy as np
from sklearn.decomposition import PCA

# Set OpenMP to use a single thread
#os.environ['OMP_NUM_THREADS'] = '1'

# Create a small random dataset
data = np.random.rand(45, 768)

# Check data integrity
print(type(data), data.shape)

# Reduce embeddings with UMAP
print("UMAP")
reducer = umap.UMAP(n_neighbors=6, n_components=10, metric='cosine')
reduced_embeddings = reducer.fit_transform(data)
print(type(reduced_embeddings), reduced_embeddings.shape)
print(reduced_embeddings)

print("PCA")
pca = PCA(n_components=10)
reduced_embeddings = pca.fit_transform(data)
print(type(reduced_embeddings), reduced_embeddings.shape)
print(reduced_embeddings)
