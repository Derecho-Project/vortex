import faiss
from gist_process_utils import *


# Function to build FAISS index (KMeans and IndexFlatL2)
def build_faiss_index(base, ncentroids, niter=20):
    dimension = base.shape[1] 
    print("dimension", dimension)
    kmeans = faiss.Kmeans(dimension, ncentroids, niter=niter, verbose=True)
    kmeans.train(base)
    print(f"Centroids shape: {kmeans.centroids.shape}")
    D, I = kmeans.index.search(base, 1)
    return kmeans.centroids, I, faiss.IndexFlatL2(dimension)


# Function to search the FAISS index for nearest neighbors
def search_faiss_index(index, query, k=5):
    distances, indices = index.search(query, k)
    # Print the search results
    print("Indices of nearest neighbors:", indices)
    print("Distances to nearest neighbors:", distances)
    return distances, indices



def faiss_knn_cluster(args):
    data_dir = args.embeddings_loc
    ncentroids = args.ncentroids
    niter = args.niter
    # Read base embeddings
    base_path = os.path.join(data_dir, 'gist_base.fvecs')
    base = fvecs_read(base_path)
    print("Base shape", base.shape)

    # Run KNN
    centroids, I, index = build_faiss_index(base, ncentroids, niter)
    # distances, indices = search_faiss_index(index, query[0].reshape(1, -1), k=5)

    # Write centroids embeddings
    with open(f'{data_dir}/centroids.pkl', 'wb') as file:
        pickle.dump(centroids, file)

    # Write cluster embeddings
    write_cluster_embeddings(I, base, ncentroids, data_dir)

    # Write query embeddings and ground truth
    write_query_results(data_dir)


# def main():
#     parser = argparse.ArgumentParser(description='Run FAISS clustering and save embeddings.')
#     parser.add_argument('--embeddings_loc', type=str, default='./gist', help='Directory to save embeddings and related files(default: ./gist)')
#     parser.add_argument('--ncentroids', type=int, default=3, help='Number of centroids for KMeans clustering(default: 3)')
#     parser.add_argument('--niter', type=int, default=20, help='Number of iterations for KMeans clustering(default: 20)')
#     args = parser.parse_args()

#     faiss_knn_cluster(args.embeddings_loc, args.ncentroids, args.niter)
    


# if __name__ == '__main__':
#     main()