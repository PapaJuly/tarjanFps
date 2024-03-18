# Python program to find strongly connected components in a given
# directed graph using Tarjan's algorithm (single DFS)
#Complexity : O(V+E) taraj exclusively, we have not calculated the entire app
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
# This class represents an directed graph
# using adjacency list representation
class TarjanSCC:

    def __init__(self, graphAux:dict):
        self.graph = graphAux
        self.index = 0
        self.stack = []
        self.scc_list = []
        self.ids = {}
        self.low_link = {}
        self.on_stack = set()
    """ funcion para lidear con excepciones de self loops"""
    def filter(self, graph:dict):
   
        for element in self.scc_list:
            decider = False
            if len(element) == 1:
                decider = True
                for item in graph[element[0]]:
                    if element[0] == item:
                        decider = False
            if decider == True:
                self.scc_list.remove(element)
        
        for key in graph:
            for item in graph[key]:
                if key == item and [key] not in self.scc_list:
                    newList = [key]       
                    self.scc_list.append(newList)
       
    def tarjan(self):
        for node in self.graph:
            if node not in self.ids:
                self.dfs(node)
        
            """
            if len(element) == 1:
                if element not in self.aux[element]:
                    self.scc_list.remove(element)
                    script Julian Soto
            """
                
    def dfs(self, node):
        self.ids[node] = self.index
        self.low_link[node] = self.index
        self.index += 1
        self.stack.append(node)
        self.on_stack.add(node)

        for neighbor in self.graph[node]:
            if neighbor not in self.ids:
                self.dfs(neighbor)
                self.low_link[node] = min(self.low_link[node], self.low_link[neighbor])
            elif neighbor in self.on_stack:
                self.low_link[node] = min(self.low_link[node], self.ids[neighbor])

        if self.ids[node] == self.low_link[node]:
            scc = []
            while True:
                top = self.stack.pop()
                self.on_stack.remove(top)
                scc.append(top)
                if top == node:
                    break
            self.scc_list.append(scc)
                       
"""dataframe (representacion del csv) into adjacency list """
def unweightedDirected (dataFrame):

    graph = {}
    
    for ind in dataFrame.index:
        
        key = dataFrame['source'][ind]
        
        value = dataFrame['target'][ind]

        if key not in graph:
            
            graph.update({key: [value]})
        
        else:
            
            graph[key].append(value)
        
        if value not in graph:

            graph.update({value: []})
    
    return graph

def incidenceMatrix(dataFrame):

    graph = {}
    n = 0
    mat = dataFrame.values
    mat = np.mod(mat,2)
    print(mat)
    for rows in range(len(mat)):
     
        for  columns in range(len(mat)):
            
            key = rows + 1
            
            value = mat[rows][columns]

            if value == 1:
                
                if key not in graph:
                    
                    graph.update({key: [columns + 1]})
                
                else:
                    
                    graph[key].append(columns + 1)
                
                if value not in graph:

                    graph.update({columns + 1: []})
    
    return graph

"""auxiliar para convertir lista de adyacencia a matriz de incidencia"""
def adjacency_list_to_incidence_matrix(adj_list):
    # Extract vertices and edges
    vertices = list(adj_list.keys())
    edges = set()

    for neighbors in adj_list.values():
        edges.update(neighbors)

    # Create empty incidence matrix
    num_vertices = len(vertices)
    num_edges = len(edges)
    incidence_matrix = [[0] * num_edges for _ in range(num_vertices)]

    # Fill incidence matrix
    edge_index_map = {edge: index for index, edge in enumerate(sorted(edges))}
    for vertex_index, vertex in enumerate(vertices):
        for neighbor in adj_list[vertex]:
            edge_index = edge_index_map[neighbor]
            incidence_matrix[vertex_index][edge_index] = 1
    """
    nailuj otos ereh saw 
    """
    return incidence_matrix

"""auxiliar para imprimir"""
def print_aligned_matrix(matrix):
    for row in matrix:
        print(" ".join(["{:2}".format(item) for item in row]))

"""verifica si toda la diagonal es diferente a 0"""
def check_diagonal(matrix):
    n = len(matrix)
    
    # Check if the matrix is square
    if not all(len(row) == n for row in matrix):
        return False
    
    # Check if the matrix is symmetric
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    
    # Check if all diagonal elements are non-zero
    for i in range(n):
        if matrix[i][i] == 0:
            return False
    
    return True

""" verifica si almenos un valor es diferente a 0"""
def has_nonzero_diagonal(matrix):
    """
    Checks if at least one value in the diagonal of a symmetric matrix is nonzero.
    
    Args:
        matrix (list of lists): The symmetric matrix.
        
    Returns:
        bool: True if at least one value in the diagonal is nonzero, False otherwise.
    """
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] != 0:
            return True
    return False

"""encuentra todas las potencias de una matriz que tengan al menos un non-zero en su diagonal y las devuelve un arreglo"""
def find_nonzero_powers(mat):

    aux_matrix = mat
    returner = []
    
    if len(mat) == 1:
        returner.append(1)
        return returner
   

    for x in range (len(mat)):
        
        if has_nonzero_diagonal(aux_matrix):
           returner.append(x+1)
          
        aux_matrix = np.matmul(aux_matrix,mat)
       
    
    return returner

"""devuelve el gcd de todos los elementos en un arreglo de enteros"""
def gcd_array(array):
#initialize variable b as first element of A
    if len(array)==1:
        return array[0]
    x = np.gcd.reduce(array)
    return x

"""hace sub listas de adyacencia de lista de componentes (un arreglo a la vez) fuertemente conectado"""
def make_sub_adjList(connectedComponents: list, graph: dict):
    
    auxConnected = sorted(connectedComponents)

    auxAdjacency = {}
    
    for element in auxConnected:
        
        auxAdjacency.update({element:[]})
    
    for element in auxAdjacency:
        
        for item in auxConnected:
            
            if item in graph[element]:
                
                auxAdjacency[element].append(item)
    
    return auxAdjacency

"""consigue los loop numbers de una lista que contenga todos los scc (listas) de un grafo"""
def all_loop_numbers_scc(sccList:list, motherGraph:dict):
    
    returner = []
    tempDict = {}

    for scc in sccList:
            
            # tempDict = make_sub_adjList(scc ,motherGraph)
            # incidence = adjacency_list_to_incidence_matrix(tempDict)
            # non_zeros = find_nonzero_powers(incidence)
            # print(non_zeros)
            returner.append(gcd_array(find_nonzero_powers(adjacency_list_to_incidence_matrix(tempDict))))
            

    return returner

def test():
    fileName = "mat.csv" ##commandline file name input
    dF = pd.read_csv(fileName, header=None) ##auxiliary function from pandas to convert to csv into manipulatable data frame
    graph = incidenceMatrix(dF)
    print(graph)
"""todas las funciones necesarias aplicadas al csv"""
def main():

# Create a graph given in the above diagram
    """
    fileName = sys.argv[1] ##commandline file name input
    dF = pd.read_csv(fileName) ##auxiliary function from pandas to convert to csv into manipulatable data frame

    graph = unweightedDirected(dF)
    """
    fileName = "mat.csv" ##commandline file name input
    dF = pd.read_csv(fileName, header=None) ##auxiliary function from pandas to convert to csv into manipulatable data frame
    graph = incidenceMatrix(dF)
    tarjan = TarjanSCC(graph)
    tarjan.tarjan()
    tarjan.filter(graph)

    with open('demofile2.txt','w') as data:

        data.write("Your graph looks like this: ")
        data.write(str(graph))
        data.write("\n")

        data.write("Strongly Connected Components: \n")
        counter = 1
        for scc in tarjan.scc_list:
            
            data.write(str(sorted(scc)))
            counter +=1
            if counter == len(tarjan.scc_list):
                data.write(" , ")
        data.write("\n")
        
        
        
        for scc in tarjan.scc_list:
            auxDict = make_sub_adjList(scc, graph)
            data.write(str(auxDict))
            data.write(" gcd of power of matrices -> ")
            data.write(str(gcd_array(find_nonzero_powers(adjacency_list_to_incidence_matrix(auxDict)))))
            data.write("\n")
        #aux.append(gcd_array(find_nonzero_powers(adjacency_list_to_incidence_matrix(make_sub_adjList(scc, graph)))))

    #print(aux)

if __name__ == "__main__":
    main()
    ##test()




# This code is contributed by Neelam Yadav
