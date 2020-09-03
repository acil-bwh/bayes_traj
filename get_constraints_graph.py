import numpy as np
import networkx as nx
from tqdm import tqdm

import pdb

def get_constraints_graph(ids, pairs, constraint_type, graph=None):
    """Creates (or augments) a graph with edges between nodes which are indexed
    by the array index of 'ids'. 'pairs' is a 2D tuple of IDs between which
    a constraint is to be established. Graph edges are given an attribute
    called 'constraint' that is set to 'constraint_type'. This function is
    intended to produce a constraints graph that can be used with a
    'MultDPRegression' object.
    
    Parameters
    ----------
    ids : 1D array
        Array of (subject) IDs.

    pairs : 2D tuple
        Each tuple entry should correspond to a subject ID (as stored in
        'ids').

    constraint_type : string
        One of 'weighted_must_link', 'weighted_cannot_link', 'must_link', 
        'cannot_link', or 'longitudinal'

    graph : networkx graph, optional
        If provided, this function will augment it with additional edges. If
        none specified, one will be created.
        
    Returns
    -------
    graph : networkx graph
        Node IDs correspond to 'ids' indices and paths (set of graph edges) that
        link those nodes. Graph edges are given an attribute called
        'constraint' that is set to 'constraint_type'.    
    """
    assert constraint_type == 'must_link' or constraint_type == 'cannot_link' \
      or constraint_type == 'weighted_must_link' or \
      constraint_type == 'weighted_cannot_link' \
      or constraint_type == 'longitudinal', 'Invalid constraint_type'
    
    if graph is None:
        graph = nx.Graph()

    for p in tqdm(pairs):
        ids_p1 = np.where(ids == p[0])[0]
        ids_p2 = np.where(ids == p[1])[0]
        if ids_p1.shape[0] > 0 and ids_p2.shape[0] > 0: 
            for id1 in ids_p1:
                for id2 in ids_p2:
                    graph.add_edge(id1, id2, constraint=constraint_type)
    return graph
