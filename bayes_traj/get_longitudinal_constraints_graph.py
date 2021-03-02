import numpy as np
import networkx as nx

import pdb

def get_longitudinal_constraints_graph(ids):
    """Creates a graph such that node IDs correspond to 'ids' indices and paths
    (set of graph edges) link nodes that correspond to repeated entries in
    'ids'. It is assumed that multiple entries in 'ids' correspond to multiple
    (longitudinal) visits by the same subject. Graph edges are given an
    attribute called 'constraint' that is set to 'longitudinal'. This function
    is intended to produce a contraints graph that can be used with a
    'MultDPRegression' object.
    
    Parameters
    ----------
    ids : 1D array
        Array of (subject) IDs.

    Returns
    -------
    graph : networkx graph
        Node IDs correspond to 'ids' indices and paths (set of graph edges) link
        nodes that correspond to repeated entries in 'ids'. It is assumed that
        multiple entries in 'ids' correspond to multiple (longitudinal) visits
        by the same subject. Graph edges are given an attribute called
        'constraint' that is set to 'longitudinal'.
    
    """

    unique_ids = set(ids)

    graph = nx.Graph()
    for u in unique_ids:
        node_ids = np.where(ids == u)[0]
        if node_ids.shape[0] > 1:
            for n in range(1, node_ids.shape[0]):
                graph.add_edge(node_ids[n-1], node_ids[n],
                               constraint='longitudinal')
        
    return graph
