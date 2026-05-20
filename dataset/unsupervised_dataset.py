import os
import pickle

import torch
import torch_geometric
import numpy as np

from utils import get_a_new2, extract_raw_ilp


class UnsupervisedGraphDataset(torch_geometric.data.Dataset):
    """
    Dataset for unsupervised ALM training.
    Loads from .lp/.mps instance files and returns both:
      - GNN graph features (bipartite graph)
      - Raw ILP data (A, b, c) for ALM loss computation

    Caches processed data to disk for fast subsequent loads.
    """

    def __init__(self, instance_files, cache_dir=None):
        """
        Args:
            instance_files: list of paths to .lp/.mps instance files
            cache_dir: directory to cache processed data (default: alongside instances)
        """
        super().__init__(root=None, transform=None, pre_transform=None)
        self.instance_files = instance_files
        self.cache_dir = cache_dir

    def len(self):
        return len(self.instance_files)

    def _cache_path(self, ins_path):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(ins_path))[0]
            return os.path.join(self.cache_dir, f'{base}.unsup_cache')
        return ins_path + '.unsup_cache'

    def get(self, index):
        ins_path = self.instance_files[index]
        cache_path = self._cache_path(ins_path)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            return cached

        # Extract GNN graph features
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)

        # Extract raw ILP data
        raw_ilp = extract_raw_ilp(ins_path)

        constraint_features = c_nodes
        edge_indices = A._indices()
        variable_features = v_nodes
        edge_features = torch.ones(A._values().unsqueeze(1).shape)

        constraint_features[torch.isnan(constraint_features)] = 1

        graph = UnsupervisedBipartiteData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            torch.FloatTensor(variable_features.cpu()),
        )

        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.ntcons = constraint_features.shape[0]
        graph.n_constraints = constraint_features.shape[0]

        # Raw ILP data for ALM loss
        graph.obj_coeffs = raw_ilp['obj_coeffs']
        graph.raw_cons_indices = raw_ilp['cons_indices']
        graph.raw_cons_values = raw_ilp['cons_values']
        graph.raw_rhs = raw_ilp['rhs']
        graph.raw_n_cons = raw_ilp['n_cons']
        graph.b_vars_mask = raw_ilp['b_vars_mask']
        graph.obj_sense_min = torch.tensor([1 if raw_ilp['obj_sense_min'] else 0], dtype=torch.long)

        # Variable mapping: from GNN output order to raw ILP order
        all_varname = list(v_map)
        raw_var_names = raw_ilp['var_names']
        raw_name2idx = {name: i for i, name in enumerate(raw_var_names)}

        # gnn_to_raw_map[i] = index in raw ILP of the i-th GNN variable
        gnn_to_raw_map = torch.zeros(variable_features.shape[0], dtype=torch.long)
        for gnn_idx, vname in enumerate(all_varname):
            if vname in raw_name2idx:
                gnn_to_raw_map[gnn_idx] = raw_name2idx[vname]
            else:
                gnn_to_raw_map[gnn_idx] = gnn_idx

        graph.gnn_to_raw_map = gnn_to_raw_map

        # b_vars indices (for GNN output)
        graph.b_vars = b_vars.long()

        # Store v_map keys for variable name alignment
        graph.varNames = all_varname

        # Cache to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(graph, f)
        except Exception:
            pass

        return graph


class UnsupervisedBipartiteData(torch_geometric.data.Data):
    """Bipartite graph data for unsupervised training."""

    def __init__(self, constraint_features=None, edge_indices=None, edge_features=None, variable_features=None):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == 'raw_cons_indices':
            return torch.tensor(
                [[self.raw_n_cons], [self.variable_features.size(0)]]
            )
        elif key == 'gnn_to_raw_map':
            return self.variable_features.size(0)
        elif key == 'b_vars':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'raw_cons_indices':
            return 1  # concatenate along edge dimension (dim=1)
        return super().__cat_dim__(key, value, *args, **kwargs)
