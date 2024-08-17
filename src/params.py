# composition params
c_zero = 0.3
#c_zero = 1.0
c_noise = 1e-3
seed = 3009


TEST_NAME = "jokisaari"

# physical params
T = 600.0 # K
M_c = 10**(-2257.7/T) * 10**(-4.7104) # m^2/(s)
l =  1.0e-8 # m
delta = l # m
tau = l**2/M_c # s
theta = 72 # deg

# time params
dt = 1.0
t_start = 0
t_end = 1e4
save_every = [1e2, 1e3, 1e4]
#t_end = 1e4
#save_every = [100, 1000, 10000] #[1, 5, 10, 20, 100, 200, 500, 1000, 2000, 3000, 10000]
eval_every = dt

# solver params
#kappa = M_c * tau / l**4 * l**2 # [-]
# M = 0.5
# a = 1.0
# A = 1.0
# kappa = 1.0
M = 5.0
a=0.0
A = 5.0
kappa = 2.0
alpha = 5.0

# output params and flags
DIFFUSE = True
RESTART = False
SPINODAL = False
PATH = "../output/"
FILENAME = f"time"
EVAL = False
SUBSTRATE = False
substrate_dim = 0.0
# ONLY FOR CONFIG_TYPE = NANOWIRE
FINITE = False
FINITE_LENGTH = 300
# ONLY FOR CONFIG_TYPE = SPINODAL
dx_spinodal = 1.0