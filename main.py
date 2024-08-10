import numpy as np
import tiles as t
import networkx as nx


# Define the number of vertices for the n-cycle
n = 3 # Change this value for different cycle sizes

# Create the n-cycle adjacency matrix
first = t.create_n_cycle(n)
second = t.add_row_and_column(first)

#print(f"Adjacency Matrix for an undirected {n}-cycle:")
#print(first)

#print("////")

twotile = np.array([[0., 1., 1., 1.],
                   [1., 0., 1., 1.],
                   [1., 1., 0., 0.],
                   [1., 1., 0., 0.]])

test = t.add_row_and_column(twotile)

matrices = t.add_edges(test)
#t.print_matrices(matrices)

final = t.filter_isomorphic_matrices(matrices)

bigNodes = t.find_big_nodes(twotile)

print(t.find_adjacent_nodes(twotile,bigNodes))

