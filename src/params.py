# composition params
# c_zero = 0.3 # jokisaari test cases
c_zero = 1.0 # NW configuration c=1.0 : bulk NW
             #                  c=0.0 : void
c_noise = 1e-2
seed = 3009

k_mode = 0

# TEST_NAME = "jokisaari"
TEST_NAME = "test"

# physical params
T = 600.0 # K
M_c = 10**(-2257.7/T) * 10**(-4.7104) # m^2/(s)
l =  1.0e-8 # m
delta = l # m
tau = l**2/M_c # s

theta = 90 # deg

# time params
dt_min = 0.01
dt_max = 2.0
beta = 10.0
dt = 1.0
t_start = 0
t_end = 3000
# save_every = [10, 100, 500, 1000, 2000, 3000, 4000, 5000]
# save_every = [100,500, 1000, 2000, 3000, 5000, 10000]
# save_every = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000]
save_every = [250, 500, 750, 1250, 1350, 1375, 1400, 1500]
# save_every = 500
eval_every = dt

# solver params
#kappa = M_c * tau / l**4 * l**2 # [-]
# M = 0.5*2.0
# a = 1.0
# A = 1.0/2.0
# kappa = 1.0/2.0
# alpha = M/2.0

L = 1.0 # characteristic length (-)
T = 1.0 # characteristic time (-)
E = 1.0 # characteristic energy (-)

a = 1.0
M = L**5 / (E * T)
A = E / L**3
kappa = E / L
alpha = M / 2.0
# M = 5.0
# a=0.0
# A = 5.0
# kappa = 2.0
# alpha = 5.0

# output params and flags
ADAPTIVE = False
EVAL_MU = False
DIFFUSE = False
RESTART = False
SPINODAL = False
PATH = "../output/"
FILENAME = f"time"
EVAL = False
SUBSTRATE = False
SBM = False
substrate_dim = 0.0
# ONLY FOR CONFIG_TYPE = NANOWIRE
config_name = "nanowire"
PENTAGONAL = False
FINITE = False
FINITE_LENGTH = 300
# ONLY FOR CONFIG_TYPE = SPINODAL
dx_spinodal = 0.5