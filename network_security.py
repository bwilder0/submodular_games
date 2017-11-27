import random
import networkx as nx
from functools import partial 
from double_oracle import greedy
import numpy as np
    
def initial_strategies(g, k, s, T):
    '''
    Picks a random s-T path for the attacker and a random set of K edges for the defender.
    Used to initialize double oracle
    '''
    eligible = []
    for t in T:
        if t in nx.descendants(g, s):
            eligible.append(t)
    t = random.choice(eligible)
    p = nx.shortest_path(g, s, t)
    path = []
    for i in range(len(p)-1):
        path.append((p[i], p[i+1]))
    return random.sample(g.edges(), k), path

def br_independent_nsg(g, allowed_edges, x, s, T, tau):
    '''
    BRI oracle for the NSG domain. 
    '''
    import igraph
    y = np.log(1./(1 - x))
    for u,v in g.edges():
        g[u][v]['weight'] = 0
    for i, (u,v) in enumerate(allowed_edges):
        g[u][v]['weight'] = np.log(1./(1 - x[i]))
        if g[u][v]['weight'] < 0:
            print('LESS THAN ZERO', g[u][v]['weight'])
    G = igraph.Graph(directed=True)
    G.add_vertices(len(g))
    G.add_edges(g.edges())
    for i, (u,v) in enumerate(g.edges()):
        G.es[i]['weight'] = g[u][v]['weight']
    paths = G.get_shortest_paths(s, T, weights = 'weight', output = 'epath')
    real_paths = {}
    distances = {}
    for i, t in enumerate(T):
        if len(paths[i]) == 0:
            distances[t] = np.inf
            continue
        distances[t] = 0
        real_paths[t] = []
        for j in paths[i]:
            real_paths[t].append((G.es[j].source, G.es[j].target))
            distances[t] += G.es[j]['weight']
    paths = real_paths
    for v in distances:
        distances[v] = np.exp(-distances[v])
    values = [distances[v]*tau[v] for v in T]
    best_path = None
    if np.max(values) == 0:
        for v in T:
            if v in paths:
                best_path = paths[v]
        if best_path == None:
            raise Exception('all targets unreachable')
    else:
        best_path = paths[T[np.argmax(values)]]
    return best_path, np.max(values)

def FO_NSG(x, g, allowed_edges, s, T, tau):
    '''
    Computes gradients for the NSG domain, given current marginal vector x.
    '''
    best_path, val = br_independent_nsg(g, allowed_edges, x, s, T, tau)
    best_target = best_path[-1][1]
    grad = np.zeros((len(allowed_edges)))
    for i, e in enumerate(allowed_edges):
        if e in best_path:
            grad[i] = tau[best_target]*np.prod([1 - p for j,p in enumerate(x) if allowed_edges[j] in best_path and j != i])
    return grad

def payoff(cover, path, tau):
    '''
    Payoff to pure strategies where defender covers edges in cover and attacker 
    traverses the edges in path.
    '''
    if len(set(cover).intersection(path)) == 0:
        return tau[path[-1][1]]
    return 0

def expected_payoff(cover, paths, probs, tau):
    '''
    Expected payoff when the defender plays pure strategy cover and attacker
    randomizes of paths with probability probs
    '''
    return -sum([probs[i] * payoff(cover, paths[i], tau) for i in range(len(paths))])

def defender_oracle(paths, probs, k, tau, allowed_edges):
    '''
    Greedy defender best response to given attacker mixed strategy
    '''
    f = partial(expected_payoff, paths=paths, probs=probs, tau=tau)
    return greedy(allowed_edges, k, f)

def exact_defender_oracle(paths, probs, k, tau, allowed_edges):
    '''
    Exact best response oracle for the defender via MILP formulation
    '''
    import gurobipy as gp
    m = gp.Model('defender_model')
    m.params.OutputFlag = 0
    #whether each edge is selected by the defender
    edge_vars = {}
    for e in allowed_edges:
        edge_vars[e] = m.addVar(vtype = gp.GRB.BINARY, name = 'edge_' + str(e))
    #whether each path intersects with the chosen edges
    intersect_vars = []
    for j in range(len(paths)):
        intersect_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, name = 'intersect_' + str(j), lb = 0, ub = 1))    
    m.update()
    #budget constraint
    m.addConstr(gp.quicksum(edge_vars[e] for e in allowed_edges) <= k)
    #set the intersection variables
    for j in range(len(paths)):
        m.addConstr(intersect_vars[j] <= gp.quicksum(edge_vars[e] for e in paths[j] if e in allowed_edges))
    m.update()
    m.setObjective(-gp.quicksum((1 - intersect_vars[j])*probs[j]*tau[paths[j][-1][1]] for j in range(len(paths))), gp.GRB.MAXIMIZE)
    #solve
    m.optimize()
    #get the chosen edges
    cover=set()
    for e in allowed_edges:
        if edge_vars[e].x == 1:
            cover.add(e)
    return cover, m.objVal

def heuristic_attacker_oracle(covers, probs, g, s, T, tau):
    '''
    Heuristic oracle for the attacker from Jain et al 2013. Roughly follows
    Dijkstra's algorithm, with some extra logic to avoid double counting
    edge costs for edges that are covered by the same defender pure strategy.
    '''
    import numpy as np    
    queue = [s]
    caught = {}
    prev = {}
    for v in g.nodes():
        caught[v] = np.inf
    caught[s] = 0
    Xbar = {}
    Xbar[s] = covers
    while len(queue) > 0:
        u = queue[np.argmin([caught[v] for v in queue])]
        queue.remove(u)
        for v in g.successors(u):
            ce = sum([probs[i] for i in range(len(covers)) if covers[i] in Xbar[u] and (u,v) in covers[i]])
            if caught[u] + ce < caught[v]:
                caught[v] = caught[u] + ce
                prev[v] = u
                Xbar[v] = list(Xbar[u])
                for cover in Xbar[u]:
                    if (u,v) in cover:
                        Xbar[v].remove(cover)
                queue.append(v)
    payoff = [0]*len(T)
    for tnum, t in enumerate(T):
        payoff[tnum] = (1 - caught[t]) * tau[t]
    best_t = T[np.argmax(payoff)]
    path = []
    curr = best_t
    while curr != s:
        path.append((prev[curr], curr))
        curr = prev[curr]
    path = path[::-1]
    return path

def attacker_br_pure(g, cover, s, T, tau):
    '''
    Attacker's best response to a pure defender strategy of covering the edges
    in cover, calculated via Dijkstra's algorithm. Returns the best path.
    '''
    import numpy as np
    import networkx as nx
    for u,v in g.edges():
        g[u][v]['weight'] = 1
    for u,v in cover:
        g[u][v]['weight'] = np.inf
    distances, paths = nx.single_source_dijkstra(g, s, weight='weight')
    reachable_targets = [t for t in T if t in distances and distances[t] < np.inf]
    if len(reachable_targets) == 0:
        #pick an arbitrary target which is connected to the source
        best_t = T[0]
        for v in T:
            if v in distances:
                best_t = v
    else:
        best_t = reachable_targets[np.argmax([tau[t] for t in reachable_targets])]
    path = []
    for i in range(len(paths[best_t])-1):
        path.append((paths[best_t][i], paths[best_t][i+1]))
    return path
    
def mincut_fanout(g, s, T, tau, k, num_samples):
    '''
    Generates warm starts for the SNARES algorithm (a list of strategies for
    the defender and attacker)
    '''
    import random
    g = nx.DiGraph(g)
    for u,v in g.edges():
        g[u][v]['capacity'] = 1
    for v in g.successors(s):
        g[s][v]['capacity'] = np.inf
    best_t = T[np.argmax([tau[t] for t in T])]
#    min_cut = nx.minimum_edge_cut(g, s, best_t)
    part1, part2 = nx.minimum_cut(g, s, best_t)[1]
    if s in part1:
        min_cut = nx.edge_boundary(g, part1)
    else:
        min_cut = nx.edge_boundary(g, part2)
    defender_strats = []
    attacker_strats = []
    if len(min_cut) < k:
        defender_strats.append(min_cut)
        attacker_strats.append(attacker_br_pure(g, min_cut, s, T, tau))
        return defender_strats, attacker_strats
    for i in range(num_samples):
        defender_strats.append(random.sample(min_cut, k))
        attacker_strats.append(attacker_br_pure(g, defender_strats[-1], s, T, tau))
    return defender_strats, attacker_strats
        

def attacker_oracle(covers, probs, g, s, T, tau):
    '''
    Attacker best response via MILP formulation
    '''
    import gurobipy as gp
    import numpy as np
    vals = [0]*len(T)
    paths = [0]*len(T)
    for tnum, t in enumerate(T):
        m = gp.Model('adversary_model')
        m.params.OutputFlag = 0
        edge_vars = {}
        #variables giving whether each edge is chosen
        for e in g.edges():
            edge_vars[e] =  m.addVar(vtype = gp.GRB.BINARY, name = 'edge_' + str(e))
        #variables giving whether the chosen path intersects with each defender strategy
        intersect_vars = []
        for i in range(len(covers)):
            intersect_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, name = 'cover_' + str(i), lb = 0, ub = 1))
        m.update()        
        #flow conservation constraints
        m.addConstr(gp.quicksum(edge_vars[(s, v)] for v in g.successors(s)) == 1)
        for v in g.nodes():
            if v != s and v != t:
                m.addConstr(gp.quicksum(edge_vars[(u,v)] for u in g.predecessors(v)) == gp.quicksum(edge_vars[(v,u)] for u in g.successors(v)))
                m.addConstr(gp.quicksum(edge_vars[(u,v)] for u in g.predecessors(v)) <= 1)
        m.addConstr(gp.quicksum(edge_vars[(v, t)] for v in g.predecessors(t)) == 1)
        m.addConstr(gp.quicksum(edge_vars[(t, v)] for v in g.successors(t)) == 0)
        #set intersection variables
        for i in range(len(covers)):
            for e in covers[i]:
                m.addConstr(intersect_vars[i] >= edge_vars[e])
        #objective
        m.setObjective(tau[t] * gp.quicksum(probs[i]*(1 - intersect_vars[i]) for i in range(len(covers))), gp.GRB.MAXIMIZE)
        m.update()        
        #solve
        m.optimize()
        #this target isn't reachable from the sources
        if m.STATUS == gp.GRB.INFEASIBLE:
            vals[tnum] = -np.inf
            paths[tnum] = -1
            continue
        #retrieve objective value and chosen path
        vals[tnum] = m.objVal
#        paths[tnum] = [e for e in g.edges() if edge_vars[e].x == 1]
        paths[tnum] = []
        curr_node = s
        while curr_node != t:
            next_node = [v for v in g.successors(curr_node) if edge_vars[(curr_node, v)].x == 1][0]
            paths[tnum].append((curr_node, next_node))
            curr_node = next_node
    #return highest value path
    if np.all(vals == -np.inf):
        raise Exception('No target is reachable')
    return paths[np.argmax(vals)], np.max(vals)


def snares(g, S, T, k, tau, epsilon = 0.001):
    '''
    Implements the SNARES algorithm from Jain et al 2013.
    '''
    import numpy as np
    from double_oracle import solve_zero_sum
    from functools import partial
    import networkx as nx
    g = nx.DiGraph(g)
    n = len(g)
    allowed_edges = g.edges()
    if tau == None:
        tau = {}
        for t in T:
            tau[t] = 1
    supersource = n
    g.add_node(supersource)
    for s in S:
        g.add_edge(supersource, s)
    f_payoff = partial(payoff, tau=tau)
    max_oracle = partial(exact_defender_oracle, k = k, tau = tau, allowed_edges = allowed_edges)
    min_oracle = partial(attacker_oracle, g = g, s = supersource, T = T, tau = tau)
    max_better_oracle = partial(defender_oracle, k = k, tau = tau, allowed_edges = allowed_edges)
    min_better_oracle = partial(heuristic_attacker_oracle, g = g, s = supersource, T = T, tau = tau)   
    #initialize with mincut-fanout
    max_strats, min_strats = mincut_fanout(g, supersource, T, tau, k, 10)
    #current mixed strategy for each
    game_matrix = np.zeros((len(max_strats), len(min_strats)))
    for i, x in enumerate(max_strats):
        for j, y in enumerate(min_strats):
            game_matrix[i,j] = f_payoff(x, y)
    min_probs, max_probs = solve_zero_sum(np.transpose(game_matrix))
    curr_minmax_value = 0
    for i in range(len(max_strats)):
        for j in range(len(min_strats)):
            curr_minmax_value += max_probs[i]*min_probs[j]*game_matrix[i,j]

    #value of the maximizing player at each iteration
    vals = []
    iter_num = 0
    while True:
        iter_num += 1
        #start with approximate best responses
        max_br = max_better_oracle(min_strats, min_probs)[0]
        max_val = sum([min_probs[i]*f_payoff(max_br, y) for i,y in enumerate(min_strats)])

        min_br = min_better_oracle(max_strats, max_probs)
        min_val = sum([max_probs[i]*f_payoff(x, min_br) for i,x in enumerate(max_strats)])
        #compute exact best responses if approximate ones do not improve value        
        if np.abs(max_val - curr_minmax_value) < epsilon:
            max_br, max_val = max_oracle(min_strats, min_probs)
        if np.abs(min_val - curr_minmax_value) < epsilon:
            min_br, min_val = min_oracle(max_strats, max_probs)
        #terminate if both strategies are already in the respective lists
        if max_br in max_strats and min_br in min_strats:
            break
        #add strategies to the respective lists
        if max_br not in max_strats:
            max_strats.append(max_br)
        if min_br not in min_strats:
            min_strats.append(min_br)
        #terminate if improvement is < epsilon
        if np.abs(min_val - curr_minmax_value) < epsilon and np.abs(max_val - curr_minmax_value) < epsilon:
            break
        #compute the game matrix restricted to current strategies
        game_matrix = np.zeros((len(max_strats), len(min_strats)))
        for i, x in enumerate(max_strats):
            for j, y in enumerate(min_strats):
                game_matrix[i,j] = f_payoff(x, y)
        #get new mixed strategies
#        max_probs, min_probs = solve_zero_sum(game_matrix)
        min_probs, max_probs = solve_zero_sum(np.transpose(game_matrix))
        for i, p in enumerate(min_probs):
            if p < 0.0001:
                min_probs[i] = 0
        for i, p in enumerate(max_probs):
            if p < 0.0001:
                max_probs[i] = 0
        curr_minmax_value = 0
        for i in range(len(max_strats)):
            for j in range(len(min_strats)):
                curr_minmax_value += max_probs[i]*min_probs[j]*game_matrix[i,j]
        #record value for this iteration
        vals.append(curr_minmax_value)

    #final mixed strategies and value
    game_matrix = np.zeros((len(max_strats), len(min_strats)))
    for i, x in enumerate(max_strats):
        for j, y in enumerate(min_strats):
            game_matrix[i,j] = f_payoff(x, y)
    min_probs, max_probs = solve_zero_sum(np.transpose(game_matrix))
    curr_minmax_value = 0
    for i in range(len(max_strats)):
        for j in range(len(min_strats)):
            curr_minmax_value += max_probs[i]*min_probs[j]*game_matrix[i,j]

    #trim unused strategies    
    max_strats = [max_strats[i] for i,p in enumerate(max_probs) if p > 0.00001]
    max_probs = [p for i,p in enumerate(max_probs) if p > 0.00001]
    min_strats = [min_strats[i] for i,p in enumerate(min_probs) if p > 0.00001]
    min_probs = [p for i,p in enumerate(min_probs) if p > 0.00001]
    return curr_minmax_value, max_strats, max_probs

    
def nsg_fw(g, S, T, k, u, K, m, tau):
    '''
    Runs the Frank-Wolfe algorithm for the NSG domain. 
    
    g: input graph
    
    S: set of source nodes
    
    T: set of target nodes
    
    k: number of defender resources
    
    u: smoothing parameter for FW
    
    K: number of iterations to run FW
    
    m: number of stochastic gradient evaluations per iteration
    
    tau: dictionary containing payoff for each target
       
    Returns:
        
    x: the final marginal vector
    
    bases: a set of size k covers such that x is the uniformly weighted combination of the bases
    
    val: the value obtainable by the adversary best responding to x
    '''
    import networkx as nx
    from equator import greedy_top_k, sfw
    import numpy as np
    
    #edges which may be covered by the defender
    allowed_edges = g.edges()
    g = nx.DiGraph(g)
    n = len(g)
    
    #add a supersource to the graph
    supersource = n
    g.add_node(supersource)
    for s in S:
        g.add_edge(supersource, s)
    
    FO = partial(FO_NSG, g=g, allowed_edges = allowed_edges, s=supersource, T=T, tau=tau)
    LO = partial(greedy_top_k, elements = allowed_edges, budget = k)
    x0 = np.zeros((len(allowed_edges)))
    
    #add 0.0001 for numerical stability in shortest paths
    x0 += u + 0.0001
    x, points = sfw(x0, K, m, u, FO, LO)
    x -= u + 0.0001
    x = np.minimum(x, 1)
    
    bases  = []
    for i in range(len(points)):
        curr_base = []
        for j in range(len(points[i])):
            if points[i][j] == 1:
                curr_base.append(allowed_edges[j])
        bases.append(set(curr_base))
    val = br_independent_nsg(g, allowed_edges, x, supersource, T, tau)[1]
    return x, bases, val