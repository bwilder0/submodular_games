import numpy as np
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t
import random
cimport cython 

def construct_g_p(g):
    '''
    Converts a Networkx graph into an adjacency list in format
    required by later functions
    '''
    max_degree = max([len(g[v]) for v in g])
    G_array = np.zeros((len(g), max_degree), dtype=np.int)
    P = np.zeros((len(g), max_degree))
    G_array[:] = -1
    for v in range(len(g)):
        for i, u in enumerate(g[v].keys()):
            G_array[v, i] = u
            P[v, i] = g[v][u]
    return G_array, P

def construct_g_p_reverse(g, n_l, n_r):
    '''
    Converts a Networkx graph into a reverse adjacency list in format
    required by later functions
    '''
    degrees = []
    neighbors = []
    for v in range(n_l):
        neighbors.append([u for u in range(n_r) if v in g[u]])
        degrees.append(len(neighbors[-1]))
    max_degree = max(degrees)
    G_array = np.zeros((n_l, max_degree), dtype=np.int)
    P = np.zeros((n_l, max_degree))
    G_array[:] = -1
    for v in range(n_l):
        for i, u in enumerate(neighbors[v]):
            G_array[v, i] = u
            P[v, i] = g[u][v]
    return G_array, P


@cython.boundscheck(False) # turn off bounds-checking for entire function
def budget_objective(np.ndarray[INT_t, ndim=2] G, np.ndarray[FLOAT_t, ndim=2] P, np.ndarray[FLOAT_t, ndim=1] w, np.ndarray[FLOAT_t, ndim=1] y):
    '''
    Objective value for the budget allocation problem.
    
    G: graph (adjacency list)
    
    P: probability on each edge. 
    
    w: numpy array giving the weight of each vertex in R
    
    y: array where y[v] is amount of budget allocated to node v
    '''
    cdef float total
    cdef int v
    cdef float p_fail
    cdef int i
    for v in range(G.shape[0]):
        p_fail = 1
        for i in range(G.shape[1]):
            if G[v, i] == -1:
                break
            p_fail *= (1 - P[v, i])**(y[G[v, i]])
        total += w[v]*(1 - p_fail)
    return total

@cython.boundscheck(False) # turn off bounds-checking for entire function
def budget_objective_fast(np.ndarray[INT_t, ndim=2] G, np.ndarray[FLOAT_t, ndim=2] P, np.ndarray[FLOAT_t, ndim=1] w, np.ndarray[FLOAT_t, ndim=1] y, int n_r):
    '''
    Objective value for the budget allocation problem, but implemented more
    efficiently.
    
    G: graph (adjacency list)
    
    P: probability on each edge. 
    
    w: numpy array giving the weight of each vertex in R
    
    y: array where y[v] is amount of budget allocated to node v
    
    n_r: number of nodes on RHS
    '''
    cdef float total
    cdef int v
    cdef float p_fail
    cdef int i
    cdef int u
    cdef np.ndarray[FLOAT_t, ndim=1] prob_reach
    prob_reach = np.ones((n_r))
    for v in range(G.shape[0]):
        if y[v] > 0:
            for i in range(G.shape[1]):
                if G[v, i] == -1:
                    break
                u = G[v, i]
                prob_reach[u] *= (1 - P[v, i])**(y[v])
    prob_reach = 1 - prob_reach
    return np.dot(prob_reach, w)

def FO_budget_cython(np.ndarray[FLOAT_t, ndim=2] x, np.ndarray[INT_t, ndim=2] G, np.ndarray[FLOAT_t, ndim=2] P, np.ndarray[FLOAT_t, ndim=1] w, float adv_budget):
    '''
    Calculates gradient of the objective at fractional point x.
    
    x: fractional point as a vector. Should be reshapable into a matrix giving 
    probability of choosing copy i of node u.
    
    G: graph (adjacency list)
    
    P: probability on each edge. 
        
    w: weights for nodes in R
        
    adv_budget: side of D-norm uncertainty
    '''
    cdef np.ndarray[FLOAT_t, ndim=2] grad
    cdef int v 
    cdef float p_all_fail
    cdef int u
    cdef int i 
    cdef int j
    cdef int B
    grad = np.zeros((x.shape[0], x.shape[1]))
    B = x.shape[1]
    #process gradient entries one node at a time
    for v in range(G.shape[0]):
        p_all_fail = 1
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            for i in range(B):
                p_all_fail *= 1 - x[G[v, j],i]*P[v, j]
        for j in range(G.shape[1]):
            u = G[v, j]
            if u == -1:
                break
            for i in range(B):
                grad[u, i] += w[v]*P[v, j]*(p_all_fail/(1 - x[u,i]*P[v, j]))
    return grad.flatten()


def marginal_coverage(np.ndarray[FLOAT_t, ndim=2] x, np.ndarray[INT_t, ndim=2] G, np.ndarray[FLOAT_t, ndim=2] P, np.ndarray[FLOAT_t, ndim=1] w):
    '''
    Returns marginal probability that each RHS vertex is reached.
    '''
    cdef np.ndarray[FLOAT_t, ndim=1] probs
    cdef int v
    cdef int j
    cdef int b
    cdef int u
    probs = np.ones((G.shape[0]))
    for v in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            u = G[v, j]
            for b in range(x.shape[1]):
                probs[v] *= 1 - x[u, b]*P[v, j]
    probs = 1 - probs
    return probs
