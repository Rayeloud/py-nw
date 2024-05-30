# Spinodal decomposition

Solver results for the interfacial-driven spinodal decomposition.

The simulation parameters are the following:

```python
dt = 1.0 # timestep
dX = 1.0 # gridsize

M = 1.0 # interfacial-driven mobility
A = 1.0 # energy barrier size
kappa = 1.0 # scaled interfacial energy coefficient
alpha = 0.5 # stability parameter of the implicit euler method
```

The results are then compared with [Zhu et al.](https://doi.org/10.1103/PhysRevE.60.3564)