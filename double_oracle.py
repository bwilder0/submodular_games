import random
import networkx as nx
import numpy as np

def greedy(items, budget, f):
    '''
    Generic greedy algorithm to select budget number of items to maximize f.
    
    Employs lazy evaluation of marginal gains, which is only correct when f is submodular.
    '''
    import heapq
    upper_bounds = [(-f(set([u])), u) for u in items]    
    heapq.heapify(upper_bounds)
    starting_objective = f(set())
    S  = set()
    #greedy selection of K nodes
    while len(S) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound
        if new_val >= -upper_bounds[0][0] - 0.1:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective
    
def solve_zero_sum(game_matrix):
    '''
    Finds a minimax equilibrium of the given zero-sum game via linear programming. Returns the mixed strategy
    of the row and column player respectively.
    
    Assumes that the row player wants to maximize their payoff
    '''
    import gurobipy as gp
    import numpy as np
    game_matrix = np.array(game_matrix)
    m = gp.Model('adversary_model')
    m.params.OutputFlag = 0
    row_vars = []
    #variables giving the probability that each strategy is played by the row player
    for r in range(game_matrix.shape[0]):
        row_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, name = 'row_' + str(r), lb = 0, ub = 1))
    #the value of the game
    v = m.addVar(vtype = gp.GRB.CONTINUOUS, name = 'value')
    m.update()
    #constraint that the probabilities sum to 1
    m.addConstr(gp.quicksum(row_vars) == 1, 'row_normalized')
    #constrain v to the be the minimum expected reward over all pure column strategies
    column_constraints = []
    for i in range(game_matrix.shape[1]):
        column_constraints.append(m.addConstr(v <= gp.quicksum(game_matrix[j, i]*row_vars[j] for j in range(len(row_vars))), 'response_' + str(i)))
    #objective is to maximize the value of the game
    m.setObjective(v, gp.GRB.MAXIMIZE)
    #solve
    m.optimize()        
    #get the row player's mixed strategy
    row_mixed = []
    for r in range(game_matrix.shape[0]):
        row_mixed.append(row_vars[r].x)
    #get the column player's mixed strategy from the dual variable associated with each constraint
    column_mixed = []
    for c in range(game_matrix.shape[1]):
        column_mixed.append(column_constraints[c].getAttr('Pi'))
    return row_mixed, column_mixed

def double_oracle(max_oracle, min_oracle, start_max, start_min, payoff, epsilon, num_iterations):
    '''
    Implementation of the generic double oracle algorithm.
    
    max/min_oracle: best response oracle for the maximizating (minimizing) player
    
    start_max/min: arbitrary pure strategy for the maximizing (minimizing) player to initialize with
    
    payoff: a function returning the payoff to a strategy for the max player and min player
    
    epsilon: allows early termination if payoff for maximizing player improves by less than epsilon between
    two iterations
    
    num_iterations: run for only a fixed number of iterations (or until one of the other termination criteria is met)
    Pass np.inf if it should run to convergence
    
    '''
    #set of pure strategies for each player
    max_strats = [start_max]
    min_strats = [start_min]
    #current mixed strategy for each
    max_probs = [1]
    min_probs = [1]
    #value of the maximizing player at each iteration
    vals = []
    iter_num = 0
    curr_value = payoff(start_max, start_min)
    while True:
        if iter_num >= num_iterations:
            break
        iter_num += 1
        #get best responses
        max_br, max_val = max_oracle(min_strats, min_probs)
        min_br, min_val = min_oracle(max_strats, max_probs)
        #record value for this iteration
        vals.append(min_val)
        #terminate if both strategies are already in the respective lists
        if max_br in max_strats and min_br in min_strats:
            break
        #terminate if improvement is < epsilon
        if np.abs(max_val - curr_value) < epsilon and np.abs(min_val - curr_value) < epsilon:
            break
        #add strategies to the respective lists
        if max_br not in max_strats:
            max_strats.append(max_br)
        if min_br not in min_strats:
            min_strats.append(min_br)
        #compute the game matrix restricted to current strategies
        game_matrix = np.zeros((len(max_strats), len(min_strats)))
        curr_value = 0
        for i, x in enumerate(max_strats):
            for j, y in enumerate(min_strats):
                game_matrix[i,j] = payoff(x, y)
        #get new mixed strategies
        max_probs, min_probs = solve_zero_sum(game_matrix)
        #update value
        for i, x in enumerate(max_strats):
            for j, y in enumerate(min_strats):
                curr_value += max_probs[i]*min_probs[j]*game_matrix[i,j]
#        print(iter_num, curr_value, min_val, max_val)

    max_strats = [max_strats[i] for i,p in enumerate(max_probs) if p != 0]
    max_probs = [p for i,p in enumerate(max_probs) if p != 0]
    min_strats = [min_strats[i] for i,p in enumerate(min_probs) if p != 0]
    min_probs = [p for i,p in enumerate(min_probs) if p != 0]
    return curr_value, max_strats, max_probs
        