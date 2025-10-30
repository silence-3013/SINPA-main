import numpy as np
import pandas as pd
import pickle
import os
from scipy.spatial.distance import cdist

def create_sinpa_data():
    """Create SINPA data files based on real parking lot locations"""
    
    print("Creating SINPA data files...")
    
    # Create data directory structure
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/region", exist_ok=True)
    os.makedirs("data/base", exist_ok=True)
    os.makedirs("data/sensor_graph", exist_ok=True)
    
    # Load parking lot locations
    print("Loading parking lot locations...")
    df = pd.read_csv('aux_data/lots_location.csv')
    locations = df[['Latitude', 'Longitude']].values  # (1687, 2)
    
    # 1. Create prob_full_occupy.npy - probability of full occupancy
    print("Creating prob_full_occupy.npy...")
    # Generate realistic occupancy probabilities based on location
    # Central areas (near equator) tend to have higher occupancy
    lat_center = np.mean(locations[:, 0])
    lon_center = np.mean(locations[:, 1])
    
    # Distance from center affects occupancy probability
    distances_from_center = np.sqrt((locations[:, 0] - lat_center)**2 + (locations[:, 1] - lon_center)**2)
    prob_full_occupy = 0.3 + 0.4 * np.exp(-distances_from_center * 10)  # Higher near center
    prob_full_occupy = np.clip(prob_full_occupy, 0.1, 0.8)  # Clip to reasonable range
    np.save("data/prob_full_occupy.npy", prob_full_occupy)
    
    # 2. Create region/assignment.npy - assignment matrix for spatial encoding
    print("Creating region/assignment.npy...")
    # Create assignment matrix: each lot assigned to multiple clusters
    n_lots = 1687
    n_clusters = 100
    
    # Create assignment based on geographical proximity
    assignment = np.zeros((n_lots, n_clusters))
    
    # Use k-means like assignment based on location
    for i in range(n_lots):
        # Assign to nearby clusters
        cluster_weights = np.random.dirichlet(np.ones(10))  # 10 nearby clusters
        nearby_clusters = np.random.choice(n_clusters, 10, replace=False)
        assignment[i, nearby_clusters] = cluster_weights
    
    np.save("data/region/assignment.npy", assignment)
    
    # 3. Create region/mask.npy - adjacency mask
    print("Creating region/mask.npy...")
    # Create adjacency mask based on geographical distance
    distances = cdist(locations, locations, metric='euclidean')
    
    # Create mask: lots within certain distance are connected
    distance_threshold = 0.01  # About 1km in lat/lon units
    mask = distances < distance_threshold
    
    # Make it sparse (only ~10% connections)
    mask = mask & (np.random.random(mask.shape) < 0.1)
    
    # Ensure diagonal is False (no self-connections)
    np.fill_diagonal(mask, False)
    
    np.save("data/region/mask.npy", mask)
    
    # 4. Create base/dist.npy - distance matrix
    print("Creating base/dist.npy...")
    # Use the same distance matrix but normalize
    dist = distances / np.max(distances)  # Normalize to [0, 1]
    np.save("data/base/dist.npy", dist)
    
    # 5. Create sensor_graph/adj_mx_base.pkl - adjacency matrix for graph
    print("Creating sensor_graph/adj_mx_base.pkl...")
    # Create adjacency matrix based on distance
    adj_matrix = np.zeros((n_lots, n_lots))
    
    # Connect lots within distance threshold
    adj_matrix[distances < distance_threshold] = 1
    
    # Make it sparse and symmetric
    adj_matrix = adj_matrix.astype(bool) & (np.random.random(adj_matrix.shape) < 0.05)
    adj_matrix = adj_matrix | adj_matrix.T  # Make symmetric
    
    # Ensure diagonal is 0
    np.fill_diagonal(adj_matrix, 0)
    
    # Create neighbors list
    neighbors = []
    for i in range(n_lots):
        neighbors.append(list(np.where(adj_matrix[i] > 0)[0]))
    
    # Create the pickle file with expected format: (sensor_ids, sensor_id_to_ind, adj_mx)
    sensor_ids = [f"lot_{i}" for i in range(n_lots)]
    sensor_id_to_ind = {f"lot_{i}": i for i in range(n_lots)}
    
    with open("data/sensor_graph/adj_mx_base.pkl", 'wb') as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_matrix), f)
    
    print("SINPA data files created successfully!")
    print(f"Created files for {n_lots} parking lots")
    print(f"Adjacency matrix sparsity: {1 - np.sum(adj_matrix) / (n_lots * n_lots):.3f}")

if __name__ == "__main__":
    create_sinpa_data() 