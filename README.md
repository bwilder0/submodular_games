# Overview
This repository contains code for the paper:

Bryan Wilder. Equilibrium computation and robust optimization in zero sum games with submodular structure. AAAI Conference on Artificial Intelligence. 2018. https://arxiv.org/abs/1710.00996. 

```
@inproceedings{wilder2018equilibrium,
 author = {Wilder, Bryan},
 title = {Equilibrium computation and robust optimization in zero sum games with submodular structure},
 booktitle = {Proceedings of the 32nd AAAI Conference on Artificial Intelligence},
 year = {2018}
}
```

Included is code for the core EQUATOR algorithm, domain-specific code for network security games and robust budget allocation, and code for the double oracle algorithm. 

# Dependencies
* Gurobi, for solving LPs in the budget allocation domain
* Networkx, for general graph functions
* igraph, for fast shortest paths in the NSG domain
* Cython, for fast implementation of gradient computation in budget allocation domain
* Numpy

# Setup
The only setup needed is to compile budget_cython_fast.pyx (for the budget allocation domain). Assuming that Cython is available, run

'''
python setup_budget.py build_ext --inplace
'''
