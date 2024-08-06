# composition params
c_zero = 1.0
c_noise = 1e-3
seed = 3009

# physical params
T = 600.0 # K
M_c = 10**(-2257.7/T) * 10**(-4.7104) # m^2/(s)
l =  1.0e-8 # m
delta = l # m
tau = l**2/M_c # s

# time params
dt = 1.0
t_start = 0
t_end = 500
save_every = [500]
eval_every = 100

# solver params
M = 1.0 # [-]
A = M_c * tau / l**2 # [-]
#A = 0.25
kappa = M_c * tau / l**4 * l**2 # [-]
#kappa = 0.5
alpha = 0.5

# output params and flags
RESTART = False
SPINODAL = False
PATH = "../output/"
FILENAME = f"time"
EVAL = False
SUBSTRATE = False
substrate_dim = 0.0