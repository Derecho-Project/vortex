import numpy as np
import faiss
import pickle
import os 
from collections import defaultdict
from k_means_constrained import KMeansConstrained  # Import the balanced K-Means

EMBEDDINGS_LOC = './gist'
NCENTROIDS = 3

def fvecs_read(filename, dtype=np.float32, c_contiguous=True):
    fv = np.fromfile(filename, dtype=dtype)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

# Load your data
base = fvecs_read('./gist/gist_base.fvecs')
print("Base shape:", base.shape)

groundtruth = fvecs_read('./gist/gist_groundtruth.ivecs', np.int32)
print("Groundtruth shape:", groundtruth.shape)

query = fvecs_read('./gist/gist_query.fvecs')
print("Query shape:", query.shape)

dimension = base.shape[1]  # Assumes emb_list is a 2D array (num_embeddings, embedding_dim)
print("Dimension:", dimension)

# Balanced K-Means Clustering
niter = 20  # Increased iterations for better convergence
verbose = True
d = base.shape[1]

N = len(base)
min_size = N // NCENTROIDS
max_size = min_size + (N % NCENTROIDS > 0)

print(f"Clustering with constraints: min_size={min_size}, max_size={max_size}")

# Initialize the Balanced K-Means
kmeans = KMeansConstrained(
    n_clusters=NCENTROIDS,
    size_min=min_size,
    size_max=max_size,
    init='k-means++',
    n_init=5,
    max_iter=niter,
    verbose=verbose,
    random_state=42
)

# Fit the model
kmeans.fit(base)
centroids = kmeans.cluster_centers_
assignments = kmeans.labels_

print("Centroids shape:", centroids.shape)
print("Assignments shape:", assignments.shape)

# Assign each embedding to its cluster
clustered_embs = [[] for _ in range(NCENTROIDS)]
doc_emb_map = defaultdict(dict)

for i, cluster in enumerate(assignments):
    clustered_embs[cluster].append(base[i])
    emb_id = len(clustered_embs[cluster]) - 1
    doc_emb_map[cluster][emb_id] = i

# Create query texts
querytexts = ["Query " + str(i) for i in range(len(query))]

# Save the centroids and embeddings
os.makedirs(EMBEDDINGS_LOC, exist_ok=True)

with open(f'{EMBEDDINGS_LOC}/centroids.pkl', 'wb') as file:
    pickle.dump(centroids, file)

with open(f'{EMBEDDINGS_LOC}/embeddings_list.pkl', 'wb') as f:
    pickle.dump(base, f)

for i in range(NCENTROIDS):
    with open(f'{EMBEDDINGS_LOC}/cluster_{i}.pkl', 'wb') as f:
        pickle.dump(clustered_embs[i], f)

with open(f'{EMBEDDINGS_LOC}/doc_emb_map.pkl', 'wb') as f:
    pickle.dump(doc_emb_map, f)

# Save ground truth and queries
np.savetxt(f'{EMBEDDINGS_LOC}/groundtruth.csv', groundtruth, delimiter=",", fmt='%i')
np.savetxt(f'{EMBEDDINGS_LOC}/query.csv', querytexts, fmt="%s")
np.savetxt(f'{EMBEDDINGS_LOC}/query_emb.csv', query, delimiter=",")

# Build FAISS index for searching
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(base)

# Search for the nearest 5 neighbors
k = 5  # number of nearest neighbors
distances, indices = index.search(query[0].reshape(1, -1), k)

# Print the results
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)
print("Groundtruth:", groundtruth[0])
