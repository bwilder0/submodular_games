def f_budget(y, w, g, L, R):
    '''
    Objective value for the budget allocation problem. 
    
    y: list or array where y[v] is amount of budget allocated to node v
    
    g: representation of transmission probabilities. g[v][u] is the probability that
    a node u \in L reaches a node v \in R. Iterating over g[v] should give the neighbors
    of v.
    
    L: list of the left hand side vertices
    
    R: list of the right hand side vertices
    
    w: list or numpy array giving the weight of each vertex in R
    '''
    total = 0
    for v in R:
        p_fail = 1
        for u in g[v]:
            p_fail *= (1 - g[v][u])**(y[u])
        total += w[v]*(1 - p_fail)
    return total

def allocator_oracle(ws, wprobs, g, L, R, B):
    '''
    Best response oracle for the maximizing player, used in the double oracle
    algorithm. Runs the greedy algorithm.
    
    ws: list of weight vectors
    
    wprobs: probability that adversary plays each w vector
    
    g: represention of graph
    
    L: left vertices
    
    R: right vertices
    
    B: total budget to be allocated
    '''
    import heapq
    import numpy as np
    from budget_cython_fast import construct_g_p_reverse, budget_objective_fast
    G, P = construct_g_p_reverse(g, len(L), len(R))
    convert_ws = []
    for w in ws:
        convert_ws.append(np.array(w))
    ws = convert_ws
    f = lambda y: sum(wprobs[i] * budget_objective_fast(G, P, ws[i], y, len(R)) for i in range(len(ws)))
    y = np.zeros((len(R)))
    upper_bounds = []
    
    #add B copies of each node in R
    for u in R:
        y[u] += 1
        for b in range(B):
            upper_bounds.append((-f(y), u))
        y[u] -= 1
    #run greedy algorithm
    heapq.heapify(upper_bounds)
    starting_objective = 0
    #greedy selection of K nodes
    while y.sum() < B:
        val, u = heapq.heappop(upper_bounds)
        y[u] += 1
        new_total = f(y)
        y[u] -= 1
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound
        if new_val >= -upper_bounds[0][0] - 0.1:
            y[u] += 1
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return list(y), starting_objective

def adversary_bri_oracle(x, g, L, R, w_hat, adv_budget, G, P, m, w_vars):
    '''
    Best response of the adversary to an independent distribution.
    
    x: representation of distribution. x should be reshapable into dimensions |L| * B where x[i,j]
    is probability of selecting jth copy of vertex i of L.
    
    g: representation of the graph and transmission probabilities (see f_budget)
    
    L: list of left hand side vertices
    
    R: list of right hand side vertices
    
    w_hat: estimated weights for the vertices of R
    
    adv_budget: adversary's budget (defining the size of the uncertainty set)
    '''
    import numpy as np
    from budget_cython_fast import marginal_coverage
    x = x.reshape((len(L), len(x)/len(L)))
    #calculate probability that each vertex in R is reached
    probs = marginal_coverage(x, G, P, np.array(w_hat))
    #call adversary LP
    return adversary_br_to_probs(m, w_vars, probs, w_hat, adv_budget)

def adversary_br_oracle(ys, y_probs, g, L, R, w_hat, adv_budget):
    '''
    Best response of the adversary to an explicity represented distribution over allocations.
    
    ys: a list of lists or 2D numpy array where each element is a y vector giving the budget allocated
    to each node in L
    
    y_probs: probability with which each allocation in ys is used.
    
    g: representation of the graph and transmission probabilities (see f_budget)
    
    L: list of left hand side vertices
    
    R: list of right hand side vertices
    
    w_hat: estimated weights for the vertices of R
    
    adv_budget: adversary's budget (defining the size of the uncertainty set)
    '''
    import numpy as np
    #calculate failure probability for reaching each node
    probs = np.ones((len(ys), len(R)))
    for i,y in enumerate(ys):
        for v in R:
            for u in g[v]:
                probs[i, v] *= (1 - g[v][u])**(y[u])
    probs = probs.mean(axis = 0)
    #convert to success probability
    probs = 1 - probs
    #call adversary LP
    return adversary_br_old(probs, w_hat, adv_budget)

def adversary_br_to_probs(m, w_vars, probs, w_hat, adv_budget, lb = None):
    '''
    Calculates the adverary's best response to an arbitrary allocator strategy
    where probs[v] is the total probability that node v \in R is reached. This is
    the version used in EQUATOR (where reusing the same model object is more 
    efficient since many stochastic gradient evaluations are performed.)
    
    m: Gurobi model with the constraints already set
    
    w_vars: list of Gurobi variables for each entry of w
    
    w_hat: point estimate of w for each v \in R
    
    adv_budget: size of the D-norm uncertainty set
    '''
    import gurobipy as gp
    import numpy as np
   #objective: minimize weights collected by the probabilities
    m.setObjective(gp.quicksum(w_vars[v] * probs[v] for v in range(len(w_hat))), gp.GRB.MINIMIZE)
    m.optimize()
    #extract final value of w
    w = []
    for wvar in w_vars:
        w.append(wvar.x)
    return w, np.dot(w, probs)


def adversary_br_old(probs, w_hat, adv_budget, lb = None):
    '''
    Calculates the adverary's best response to an arbitrary allocator strategy
    where probs[v] is the total probability that node v \in R is reached. This 
    version is used in the double oracle algorithm.
    
    w_hat: point estimate of w for each v \in R
    
    adv_budget: size of the D-norm uncertainty set
    '''
    import gurobipy as gp
    import numpy as np
    #default lower bound of zero for all weights
    if lb == None:
        lb = np.zeros((len(w_hat)))
    m = gp.Model()
    m.params.OutputFlag = 0
    #variables representing final weight for each node
    w_vars = []
    #variables representing the amount by which the adversary decreases each weight
    u_vars = []
    for v in range(len(w_hat)):
        w_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, lb = lb[v], name = 'w_' + str(v)))
        u_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'u_' + str(v)))
    m.update()
    #constrain w_v in terms of u_v
    for v in range(len(w_hat)):
        m.addConstr(w_vars[v] == w_hat[v] + (lb[v] - w_hat[v])*u_vars[v])
    #total budget constraint for adversary
    m.addConstr(gp.quicksum(uvar for uvar in u_vars) <= adv_budget)
#    #objective: minimize weights collected by the probabilities
    m.setObjective(gp.quicksum(w_vars[v] * probs[v] for v in range(len(w_hat))), gp.GRB.MINIMIZE)
    m.optimize()
    #extract final value of w
    w = []
    for wvar in w_vars:
        w.append(wvar.x)
    return w, np.dot(w, probs)


def make_model(w_hat, adv_budget, lb = None):
    '''
    Makes a Gurobi model for the adversary LP which includes all of the constraints.
    
    w_hat: point estimate of w for each v \in R
    
    adv_budget: size of the D-norm uncertainty set
    '''
    import gurobipy as gp
    import numpy as np
    #default lower bound of zero for all weights
    if lb == None:
        lb = np.zeros((len(w_hat)))
    m = gp.Model()
    m.params.OutputFlag = 0
    #variables representing final weight for each node
    w_vars = []
    #variables representing the amount by which the adversary decreases each weight
    u_vars = []
    for v in range(len(w_hat)):
        w_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, lb = lb[v], name = 'w_' + str(v)))
        u_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'u_' + str(v)))
    m.update()
    #constrain w_v in terms of u_v
    for v in range(len(w_hat)):
        m.addConstr(w_vars[v] == w_hat[v] + (lb[v] - w_hat[v])*u_vars[v])
    #total budget constraint for adversary
    m.addConstr(gp.quicksum(uvar for uvar in u_vars) <= adv_budget)
    return m, w_vars


def FO_budget(x, g, L, R, w_hat, adv_budget):
    '''
    Calculates gradient of the objective at fractional point x.
    
    x: fractional point as a vector. Should be reshapable into a matrix giving 
    probability of choosing copy i of node u.
    
    g: graph
    
    w: weights for nodes in R
    
    L: left side nodes
    
    R: right side nodes
    
    w_hat: point estimate for w
    
    adv_budget: side of D-norm uncertainty
    '''
    import numpy as np
    #get w defining the minimizing objective at x
    w = adversary_bri_oracle(x, g, L, R, w_hat, adv_budget)[0]
    #calculate gradient
    x = x.reshape((len(L), len(x)/len(L)))
    grad = np.zeros(x.shape)
    B = x.shape[1]
    #process gradient entries one node at a time
    for v in R:
        p_all_fail = 1
        for u in g[v]:
            for i in range(B):
                p_all_fail *= 1 - x[u,i]*g[v][u]
        for u in g[v]:
            for i in range(B):
                grad[u, i] += w[v]*g[v][u]*(p_all_fail/(1 - x[u,i]*g[v][u]))
    return grad.flatten()   

def FO_budget_cython(x, g, L, R, w_hat, adv_budget, G, P, m, w_vars,):
    '''
    Calculates gradient of the objective at fractional point x. This is a faster
    implementation which calls Cython code to compute the gradient.
    
    x: fractional point as a vector. Should be reshapable into a matrix giving 
    probability of choosing copy i of node u.
    
    g: graph
    
    w: weights for nodes in R
    
    L: left side nodes
    
    R: right side nodes
    
    w_hat: point estimate for w
    
    adv_budget: side of D-norm uncertainty
    '''
    import numpy as np
    from budget_cython_fast import FO_budget_cython
    #get w defining the minimizing objective at x
    w = np.array(adversary_bri_oracle(x, g, L, R, w_hat, adv_budget, G, P, m, w_vars)[0])
    x = x.reshape((len(L), len(x)/len(L)))
    return FO_budget_cython(x, G, P, w, adv_budget)
    

def double_oracle_budget(g, L, R, w_hat, B, adv_budget):
    '''
    Calls the double oracle function on given budget allocation instances
    '''
    from functools import partial
    from double_oracle import double_oracle
    import numpy as np
    max_oracle = partial(allocator_oracle, g = g, L = L, R = R, B = B)
    min_oracle = partial(adversary_br_oracle, g = g, L = L, R = R, w_hat = w_hat, adv_budget = adv_budget)
    payoff = partial(f_budget, g = g, L = L, R = R)
    start_max = [0] * len(R)
    start_max[0] = B
    start_min = w_hat
    return double_oracle(max_oracle, min_oracle, start_max, start_min, payoff, 0.001, np.inf)

def fw_budget(g, L, R, w_hat, B, adv_budget, u, K, m):
    '''
    Runs the Stochastic Frank-Wolfe algorithm for the budget allocation domain. 
    
    g: input graph
    
    L: left side vertices
    
    R: right side vertices
    
    w_hat: point estimate for value of nodes in R
    
    B: total budget for allocator
    
    adv_budget: budget for adversary (side of D-norm ball)
    
    u: smoothing parameter for FW
    
    K: number of iterations to run FW
    
    m: number of stochastic gradient evaluations per iteration
           
    Returns:
        
    x: the final marginal vector
    
    bases: a set of size k covers such that x is the uniformly weighted combination of the bases
    
    val: the value obtainable by the adversary best responding to x
    '''
    from equator import greedy_top_k, sfw
    import numpy as np
    from functools import partial
    from budget_cython_fast import construct_g_p
    #define oracles and starting point
    G, P = construct_g_p(g)
    model, w_vars = make_model(w_hat, adv_budget)
    FO = partial(FO_budget_cython, g = g, L = L, R = R, w_hat = w_hat, adv_budget = adv_budget, G=G, P=P, m=model, w_vars=w_vars)
    LO = partial(greedy_top_k, elements = range(len(L)*B), budget = B)
    x0 = np.zeros((len(L)*B))
    
    #run SFW
    x, points = sfw(x0, K, m, u, FO, LO)
    x = np.minimum(x, 1)
    
    #extract the set of bases which x is a convex combination of
    bases  = []
    for i in range(len(points)):
        y = points[i]
        y = y.reshape((len(L), B))
        y = y.sum(axis=1)
        bases.append(y)
    #current worst case value
    val = adversary_bri_oracle(x, g, L, R, w_hat, adv_budget, G, P, model, w_vars,)[1]
    return x, bases, val

def swap_round_budget(bases):
    '''
    Swap rounding for the budget domain. Converts vectors y giving budget
    allocated to each node into bases (lists of items chosen) and then
    calls swap_round from equator.py
    '''
    from equator import swap_round
    import numpy as np
    alt_bases = []
    for y in bases:
        new_base = set()
        for v in range(len(y)):
            for i in range(int(y[v])):
                new_base.add((v, i))
        alt_bases.append(new_base)
    rounded = swap_round(alt_bases, [1./len(alt_bases)]*len(alt_bases))
    y_round = np.zeros((len(y)))
    for x in rounded:
        y_round[x[0]] += 1
    return y_round
        