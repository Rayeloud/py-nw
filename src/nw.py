"""
nw.py

This script sets up and runs a simulation for a phase-field model. 

The model parameters include composition parameters (initial composition, 
noise spread in composition, and seed for the random number generator), 
time parameters (time step, end time, and frequency of output), 
and solver parameters (mobility, height of the double well potential, 
and gradient energy coefficient).

The script imports necessary modules from `solver` and `processing`.
"""

from solver import *
from processing import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

# composition params
c_zero = 1.0
c_noise = 1e-3
seed = 3009
width = 200

# time params
dt = 1.0
t_start = 0
t_end = 1750
save_every = 350#500#[int(500//dt), (1500//dt), int(2700//dt), int(2800//dt)]
eval_every = int(10//dt)

# solver params
M = 1.0
A = 1.0
kappa = 1.0
alpha = 0.5

# output params and flags
RESTART = False
SPINODAL = False
PATH = "../output/"
FILENAME = f"time"
EVAL = False

if __name__ == '__main__':
    # ----------------- PREPROCESSING ----------------- #

    # get arguments
    if len(sys.argv) < 2:
        print("Usage: python3 nw.py <file.geo> [-restart <time>] [-spinodal] [-path <path>]")
        sys.exit(1)

    # find flag for post-processing or restart of sim (-pproc or -restart) in sys.argv
    if "-restart" in sys.argv:
        RESTART = True
        option = sys.argv.index("-restart")
        t_start = int(sys.argv[option+1])
    if "-spinodal" in sys.argv:
        SPINODAL = True
    if "-path" in sys.argv:
        option = sys.argv.index("-path")
        PATH = sys.argv[option+1]
    if "-eval" in sys.argv:
        EVAL = True
    if "-seed" in sys.argv:
        option = sys.argv.index("-seed")
        seed = int(sys.argv[option+1])
    if "-filename" in sys.argv:
        option = sys.argv.index("-filename")
        FILENAME = sys.argv[option+1]

    c = None

    nb_steps = int((t_end-t_start)/dt)

    if RESTART:
        # ----------------- RESTART ----------------- #
        filename = f"{PATH}{FILENAME}_{t_start:06}.vtk"
        print(f"Restarting from {filename} ...")

        c, config = readVTK(filename)

        nx, ny, nz = c.shape

        dX = config["dX"]
        dx, dy, dz = dX, dX, dX
        Lx, Ly, Lz = nx*dx, ny*dy, nz*dz


        print(f'--------------------------------------------------')
        print(f'Loaded {filename}')
        print(f'Domain size: {nx} x {ny} x {nz}')
        print(f'Configuration: nb nanowires = {config["nb_nw"]}')
        print(f'Radii: R_1 = {config["r_1"]} and R_2 = {config["r_2"]}')\
            if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]}')
        print(f'Angle: {config["angle"]}')\
            if config["nb_nw"] == 2 else print(f'No orientation specified')
        print(f'\nSolving on Device: {device}')
        print(f'--------------------------------------------------')

        # convert to torch tensor
        c = torch.tensor(c, dtype=torch.float32)

    else:
        # ----------------- PREPROCESSING ----------------- #
        if not SPINODAL:
            print(f'New simulation...')
            # Set .geo file
            config = loadGeom(sys.argv[1])

            print(f'--------------------------------------------------')
            print(f'Loaded {sys.argv[1]}')
            print(f'Domain size: {config["Nx"]} x {config["Ny"]} x {config["Nz"]}')
            print(f'Configuration: nb nanowires = {config["nb_nw"]}')
            print(f'Radii: R_1 = {config["r_1"]} and R_2 = {config["r_2"]}')\
                if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]}')
            print(f'Angle: {config["angle"]}')\
                if config["nb_nw"] == 2 else print(f'No orientation specified')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')

            FILENAME = f'{FILENAME}_R1_{int(config["r_1"])}_R2_{int(config["r_2"])}'

            # read .geo file
            nx = config["Nx"]
            ny = config["Ny"]
            nz = config["Nz"]

            Lx = config["Lx"]
            Ly = config["Ly"]
            Lz = config["Lz"]

            dX = config["dX"]
            dx, dy, dz = dX, dX, dX

            # create a composition field
            print("Preprocessing...")
            c, coords = makeCompositionFieldParallel((nx, ny, nz), (Lx, Ly, Lz), c_zero, file=sys.argv[1])

            # print time 0 frame
            saveFrame(c, t_start, save_every, config, path=PATH, filename=FILENAME)

            # add noise to the composition field
            print("Adding noise to the composition field...")
            torch.manual_seed(seed)
            random_noise = c_noise * (0.5 - torch.rand(size=c.shape, dtype=torch.float32))
            #random_noise = torch.randn(size=c.shape, dtype=torch.float32) * c_noise

            c = c + random_noise
        else:
            # spinodal decomposition simulation
            dX = 1.0
            dx, dy, dz = dX, dX, dX
            Lx, Ly, Lz = 1024, 1024, 1
            nx, ny, nz = int(Ly//dx), int(Ly//dy), int(Lz//dz)
            c_noise = 0.2
            torch.manual_seed(seed)
            c = 0.5 + c_noise * (0.5 - torch.rand((nx, ny, nz), dtype=torch.float32))
            #c = 0.5 + torch.randn((nx, ny, nz), dtype=torch.float32) * c_noise
            #c = 0.0 + c_noise * (0.5 - torch.rand((nx, ny, nz), dtype=torch.float32))
            config = {"Nx": nx, "Ny": ny, "Nz": nz, 
                      "Lx": nx*dX, "Ly": ny*dX, "Lz": nz*dX, 
                      "dX": dX, "type": "spinodal"}

            print(f'New simulation...')
            print(f'--------------------------------------------------')
            print(f'No geometry file specified')
            print(f'Domain size: {nx} x {ny} x {nz}')
            print(f'Configuration: spinodal decomposition')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')
            saveFrame(c, t_start, save_every, config, path=PATH, filename=FILENAME)

    # ----------------- SOLVER ----------------- #
    with torch.no_grad():
        # define fourier modes
        _kx = torch.fft.fftfreq(c.shape[0], d=dx) * 2.0 * np.pi
        _ky = torch.fft.fftfreq(c.shape[1], d=dy) * 2.0 * np.pi
        _kz = torch.fft.fftfreq(c.shape[2], d=dz) * 2.0 * np.pi

        kx, ky, kz = torch.meshgrid(_kx, _ky, _kz, indexing='ij')

        k2 = kx**2 + ky**2 + kz**2
        k4 = k2*k2

        # define indicator function
        radius = 50
        x1 = Lx//2
        y1 = Ly//2
        z1 = Lz//2
        _x = torch.arange(start=0.0, end=Lx, step=dx)
        _y = torch.arange(start=0.0, end=Ly, step=dy)
        _z = torch.arange(start=0.0, end=Lz, step=dz)

        xx, yy, zz = torch.meshgrid(_x, _y, _z, indexing='ij')
        # circular kernel
        #indic_device = 1/2 * (1+torch.tanh((radius-torch.sqrt((xx-x1)**2+(yy-y1)**2+(zz-z1)**2))/(np.sqrt(3)*dx)))
        # planar kernel
        indic_device = 1/2 * (1+torch.tanh(-(Ly-Ly//2-width-yy)/(np.sqrt(3)*dx)))

        # bring to device
        kx = kx.to(device=device)
        ky = ky.to(device=device)
        kz = kz.to(device=device)

        k2 = k2.to(device=device)
        k4 = k4.to(device=device)
    
        c_device = c.to(device=device)
        indic_device = indic_device.to(device=device)
        # c_device.mul_(indic_device)
        # preallocate tensors in device before the loop
        torch.set_default_device(device)

        # real space
        g_device = torch.zeros_like(c_device)
        phi_device = torch.zeros_like(c_device)
        y1_device = torch.zeros_like(c_device)
        y2_device = torch.zeros_like(c_device)
        y3_device = torch.zeros_like(c_device)

        # fourier space
        c_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        g_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        r_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        q1_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        q2_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        q3_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)

    # ----------------- TIME LOOP ----------------- #

    if EVAL:
        mass_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
        energy_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
        mass_device[0] = evalTotalMass(c_device)
        energy_device[0] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz)

    idx_eval = 1
    with torch.no_grad():
        # param = {'dt':dt, 'alpha':alpha, 'kappa':kappa, 'M':M, 'A':A, 'save_every':save_every}
        # solve_cahn_hilliard(c, config, (t_start, t_end), param, mode='RK4', path=PATH, filename=FILENAME)
        # exit()
        for t in tqdm(range(t_start+1, t_start+nb_steps+1), desc="Solving..."):
            # compute g
            compute_g(g_device, c_device, A)

            # compute phi
            compute_vm(phi_device, c_device, M)
            # phi_device.mul_(indic_device)

            g_device_fft.copy_(g_device)
            c_device_fft.copy_(c_device)

            torch.fft.fftn(g_device_fft, out=g_device_fft)
            torch.fft.fftn(c_device_fft, out=c_device_fft)

            # compute r
            compute_r(r_device_fft, c_device_fft, g_device_fft, kappa, k2)

            # compute y
            compute_y((y1_device, y2_device, y3_device), r_device_fft, kx, ky, kz)

            # compute q
            compute_q((q1_device_fft, q2_device_fft, q3_device_fft), y1_device, y2_device, y3_device, phi_device)

            # update c
            compute_euler_timestep(c_device_fft, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k4, kappa, alpha, dt)

            # bring back to real space
            c_device.copy_(torch.fft.ifftn(c_device_fft, dim=(0, 1, 2), out=c_device_fft))

            if torch.max(c_device).item() < 0.5:
                print(f'Warning! Order parameter c stricly below critical value.')

            saveFrame(c_device, t, save_every, config, path=PATH, filename=FILENAME)
            if EVAL and t % eval_every == 0:
                mass_device[idx_eval] = evalTotalMass(c_device)
                energy_device[idx_eval] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz)
                idx_eval += 1

    # ----------------- POST-PROCESSING ----------------- #
    if EVAL:
        torch.save(mass_device.cpu(), f"{PATH}mass_{FILENAME}.pt")
        torch.save(energy_device.cpu(), f"{PATH}energy_{FILENAME}.pt")
    
    # clear memory
    del kx, ky, kz, k2, k4
    del c_device, g_device, phi_device, y1_device, y2_device, y3_device
    del c_device_fft, g_device_fft, r_device_fft
    del q1_device_fft, q2_device_fft, q3_device_fft
