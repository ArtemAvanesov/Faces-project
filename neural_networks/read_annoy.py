from annoy import AnnoyIndex


def read_annoy(embedding):
    f = 64  # Embedding size
    u = AnnoyIndex(f, 'euclidean')
    u.load('./neural_networks/models/saved_annoy.ann')  # super fast, will just mmap the file
    n = 1  # Num neighbors
    neighbors = []
    distances = []
    for emb in embedding:
        neighbor, dist = u.get_nns_by_vector(emb, n, search_k=-1, include_distances=True)
        neighbors.append(neighbor)
        distances.append(dist)

    return neighbors, distances
