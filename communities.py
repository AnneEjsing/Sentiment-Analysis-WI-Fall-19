import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.cluster import KMeans
import scipy
import collections
import matplotlib.pyplot as plt

def load_friendships_data():
    try:
        f=open("data/friendships.txt", "r")
        content = f.read()
        f.close()
        return content
    except:
        return None

def parseFile(file):
    userList = []
    for review in file.split("\n\n"):
        line = review.split("\n")

        user = line[0].split("user:")[1].strip()
        friends = list(((user, friend) for friend in line[1].split('\t')[1:]))
        summary = line[2].split()[1]
        review = line[3].split()[1]
        userList.append((user, friends, summary, review))
    return userList
    

def fillGraph(userList):
    G = nx.Graph()
    for (user, friends, summary, review) in userList:
        G.add_node(user)
        G.add_edges_from(friends)
    return G

def spectral_clustering(G, num_clusters):
    # Computes Laplacian matrix
    L = laplacian_matrix(G)

    # Computes all eigen vectors and eigen values
    vals, vecs = np.linalg.eig(L)

    # Sort these based on the eigenvalues
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # Remove the first eigenvector. We need to transpose it to use it in k-means
    relevant_vectors = vecs[:, 1:num_clusters].reshape(-1, num_clusters - 1)

    # Compute split based on vector space
    componspace = KMeans(n_clusters=num_clusters).fit_predict(relevant_vectors)
    communities = compute_community_for_user(G.nodes(), componspace)
    return communities

def laplacian_matrix(G):
    A = nx.to_numpy_matrix(G)
    # Compute sums of rows and enters them to a sums vector
    sums = A.sum(axis=1)
    # Compute diagonal matrix out of sums - i.e. node degrees
    D = np.zeros((A.shape[0], A.shape[1]))
    np.fill_diagonal(D, np.array(sums.reshape(1,-1))[0])
    # Computes unnormalised graph Laplacian
    L = D - A
    return L

def scipy_spectral(G, num_clusters):
    adj_mat = nx.to_numpy_matrix(G)
    sc = SpectralClustering(num_clusters, affinity='precomputed',n_init=100).fit_predict(adj_mat)
    communities = compute_community_for_user(G.nodes(), sc)
    return communities

def compute_community_for_user(users, communities):
    predictions = list(zip(users, communities))

    communities = collections.defaultdict(list)
    for p in predictions:
        key, value = p
        if key not in communities[value]:
            communities[value].append(key)
    return communities

def create_communities(number_of_communities, type='spectral'):
    file = load_friendships_data()
    userList = parseFile(file)
    G = fillGraph(userList)
    communities = {}
    if type is 'spectral':
        communities = spectral_clustering(G, number_of_communities)
    if type is 'scipy':
        communities = scipy_spectral(G, number_of_communities)
    
    return communities

if __name__ == "__main__":
    create_communities(4)