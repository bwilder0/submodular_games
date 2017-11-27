def sfw(x0, K, m, u, FO, LO):
    '''
    Stochastic Frank-Wolfe algorithm.
    
    x0: initial point in R^n
    
    K: number of iterations
    
    m: number of stochastic gradients per iteration
    
    FO: stochastic first-order oracle (returns unbiased estimate of gradient)
    
    LO: linear optimization oracle over the feasible set
    '''
    import numpy as np
    x = np.array(x0)
    points = []
    for i in range(K):
        grad = np.zeros((len(x0)))
        for j in range(m):
            Z = np.random.uniform(low = -u, high = u, size = len(x))
            x_capped = np.minimum(x + Z, 1)
            grad += FO(x_capped)
        grad /= m
        v = LO(grad)
        x = x + (1./K)*v
        points.append(v)
    return x, points

def greedy_top_k(grad, elements, budget):
    '''
    Greedily select budget number of elements with highest weight according to
    grad
    '''
    import numpy as np
    combined = zip(elements, grad)
    combined.sort(key = lambda x: -x[1])
    indicator = np.zeros((len(elements)))
    for i in range(budget):
        indicator[elements.index(combined[i][0])] = 1
    return indicator

def swap_round(bases, weights):
    '''
    Swap rounding algorithm of Chekuri et al for rounding points in the matroid
    base polytope. Implemented here just for the uniform matroid.
    '''
    import random
    bases_copy = []
    for base in bases:
        bases_copy.append(set(base))
    bases = bases_copy
    merged_base = bases[0]
    merged_weight = weights[0]
    for t in range(1, len(bases)):
        while merged_base != bases[t]:
            i = next(iter(merged_base.difference(bases[t])))
            j = next(iter(bases[t].difference(merged_base)))
            if random.random() < merged_weight/(merged_weight + weights[t]):
                bases[t].add(i)
                bases[t].remove(j)
            else:
                merged_base.add(j)
                merged_base.remove(i)
        merged_weight += weights[t]
    return merged_base