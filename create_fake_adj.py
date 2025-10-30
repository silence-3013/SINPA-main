import numpy as np
import pickle
import os

os.makedirs('data/sensor_graph', exist_ok=True)
n_nodes = 1687
adj = np.eye(n_nodes, dtype=bool)
sensor_ids = ['lot_' + str(i) for i in range(n_nodes)]
sensor_id_to_ind = {'lot_' + str(i): i for i in range(n_nodes)}
with open('data/sensor_graph/adj_mx_base.pkl', 'wb') as f:
    pickle.dump((sensor_ids, sensor_id_to_ind, adj), f)
print('Fake adj_mx_base.pkl created.') 