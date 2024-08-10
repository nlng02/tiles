import numpy as np
import itertools 
import networkx as nx

def create_n_cycle(n):
    # Initialize an n x n matrix with zeros
    matrix = np.zeros((n, n), dtype=int)
    
    # Set edges to form an undirected cycle
    for i in range(n):
        matrix[i, (i + 1) % n] = 1
        matrix[(i + 1) % n, i] = 1
    
    return matrix

def add_row_and_column(matrix):
    input_dimension = matrix.shape[0]

    zeros_row = np.zeros(input_dimension)
    zeros_column = np.zeros((input_dimension+1,1))

    matrix = np.vstack([matrix, zeros_row])
    matrix = np.hstack([matrix, zeros_column])

    return matrix


def find_pairs_of_vertices(n):
    # Generate all combinations with replacement, not including the last vertex
    combinations = list(itertools.combinations_with_replacement(range(n-1), 2))
    
    # Filter out tuples where both elements are the same
    filtered_combinations = [combo for combo in combinations if combo[0] != combo[1]]
    
    return filtered_combinations

def add_edges(matrix):
    
    indices = find_pairs_of_vertices(matrix.shape[0])
    matrices = []

    for index_pair in indices:    
        matrix_copy = matrix.copy()
        matrix_copy[index_pair[0], -1] = 1
        matrix_copy[index_pair[1], -1] = 1
        matrix_copy[-1, index_pair[0]] = 1
        matrix_copy[-1, index_pair[1]] = 1
        matrices.append(matrix_copy)

    return matrices
        

def print_matrices(matrices):
    for i, matrix in enumerate(matrices):
        print(f"Matrix {i+1}:\n{matrix}\n")

def are_isomorphic(adj_matrix1, adj_matrix2):
    """
    Check if two adjacency matrices represent isomorphic graphs.
    
    Parameters:
    adj_matrix1 (numpy.ndarray): The adjacency matrix of the first graph.
    adj_matrix2 (numpy.ndarray): The adjacency matrix of the second graph.
    
    Returns:
    bool: True if the graphs are isomorphic, False otherwise.
    """
    G1 = nx.from_numpy_array(adj_matrix1)
    G2 = nx.from_numpy_array(adj_matrix2)
    return nx.is_isomorphic(G1, G2)

def filter_isomorphic_matrices(matrices):
    filtered_matrices = []
    seen = []

    for matrix in matrices:
        is_unique = True
        for seen_matrix in seen:
            if are_isomorphic(matrix, seen_matrix):
                is_unique = False
                break
        if is_unique:
            filtered_matrices.append(matrix)
            seen.append(matrix)
    
    return filtered_matrices

def find_degrees_of_nodes(matrix):
    G = nx.from_numpy_array(matrix)
    return G.degree

def find_big_nodes(matrix):
    bigNodes = []
    nodesDegrees = find_degrees_of_nodes(matrix)
    for nodeDegree in nodesDegrees:
        if nodeDegree[1] >=3:
            bigNodes.append(nodeDegree[0])
    return bigNodes
    
def find_adjacent_nodes(matrix, nodes):
    adjacent_nodes = []
    for i in range(len(nodes)):
        firstNode = nodes[i]
        for j in range(i+1,len(nodes)):
            secondNode = nodes[j]
            if matrix[firstNode][secondNode] == 1:
                    adjacent_nodes.append([firstNode,secondNode])
    return adjacent_nodes
