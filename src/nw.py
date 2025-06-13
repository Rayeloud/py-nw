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
from params import *
from utils import *

#device='cpu'

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

if __name__ == '__main__':
    # ----------------- PREPROCESSING ----------------- #

    # get arguments
    if len(sys.argv) < 2:
        print("Usage: python3 nw.py <file.geo> [-restart <time>] [-spinodal] [-path <path>]")
        sys.exit(1)

    # find flag in sys.argv
    if "-restart" in sys.argv:
        RESTART = True
        option = sys.argv.index("-restart")
        t_start = float(sys.argv[option+1])
    if "-spinodal" in sys.argv:
        SPINODAL = True
        config_name = "spinodal"
    if "-path" in sys.argv:
        option = sys.argv.index("-path")
        PATH = sys.argv[option+1]
    if "-eval" in sys.argv:
        EVAL = True
    if "-eval_mu" in sys.argv:
        EVAL_MU = True
    if "-seed" in sys.argv:
        option = sys.argv.index("-seed")
        seed = int(sys.argv[option+1])
    if "-filename" in sys.argv:
        option = sys.argv.index("-filename")
        FILENAME = sys.argv[option+1]
    if "-substrate" in sys.argv:
        option = sys.argv.index("-substrate")
        SUBSTRATE = True
        substrate_dim = float(sys.argv[option+1])
    if "-sbm" in sys.argv:
        option = sys.argv.index("-sbm")
        SBM = True
        substrate_dim = float(sys.argv[option+1])
    if "-dt" in sys.argv:
        option = sys.argv.index("-dt")
        dt = float(sys.argv[option+1])
    if "-t_end" in sys.argv:
        option = sys.argv.index("-t_end")
        t_end = float(sys.argv[option+1])
    if "-dx" in sys.argv:
        option = sys.argv.index("-dx")
        dx_spinodal = float(sys.argv[option+1])
    if "-alpha" in sys.argv:
        option = sys.argv.index("-alpha")
        alpha = float(sys.argv[option+1])
    if "-finite" in sys.argv:
        option = sys.argv.index("-finite")
        FINITE = True
        FINITE_LENGTH = float(sys.argv[option+1])
    if "-diffuse" in sys.argv:
        option = sys.argv.index("-diffuse")
        DIFFUSE = bool(sys.argv[option+1])
    if "-E" in sys.argv:
        option = sys.argv.index("-E")
        E = float(sys.argv[option+1])
    if "-T" in sys.argv:
        option = sys.argv.index("-T")
        T = float(sys.argv[option+1])
    if "-L" in sys.argv:
        option = sys.argv.index("-L")
        L = float(sys.argv[option+1])
    if "-alpha" in sys.argv:
        option = sys.argv.index("-alpha")
        alpha = float(sys.argv[option+1])
    if "-k" in sys.argv:
        option = sys.argv.index("-k")
        k_mode = int(sys.argv[option+1])

    c = None

    t_end = t_end *T
    save_every = np.array(save_every) * T
    save_every = int(t_end // 8)

    M = L**5 / (E * T)
    A = E / L**3
    kappa = E / L
    alpha = M / 2.0 # solution to optimization problem MAX(1/2(MAX(M(c) - MIN(M(c))) = M/2.0
    eval_every = dt

    # save_every = [t_end/2, t_end]
    # if isinstance(save_every, list):
    #     for i, freq in enumerate(save_every):
    #         save_every[i] = int(freq/dt)
    # else:
    #     save_every = int(save_every/dt)
    # eval_every = int(eval_every/dt)

    nb_steps = int((t_end-t_start)/dt)

    theta = theta * np.pi/180.0

    # some print in .log file
    with open(f"{PATH}{FILENAME}_param.log", "w") as f:
        print(f'--------------------------------------------------', file=f)
        print(f'Physical parameters:', file=f)
        print(f'Mobility: M_c = {M_c} m^2/s', file=f)
        print(f'Temperature: T = {T} K', file=f)
        print(f'Interfacial width: delta = {delta} m', file=f)
        print(f'Length scale: l = {l} m', file=f)
        print(f'Time scale: tau = {tau} s', file=f)
        print(f'--------------------------------------------------', file=f)

        print(f'--------------------------------------------------', file=f)
        print(f'Non-dimensionalised parameters:', file=f)
        print(f'Characteristic length: L = {L}', file=f)
        print(f'Characteristic energy: E = {E}', file=f)
        print(f'Characteristic time: T = {T}', file=f)
        print(f'Mobility: M = {M}', file=f)
        print(f'Height of the double well potential: A = {A}', file=f)
        print(f'Gradient energy coefficient: kappa = {kappa}', file=f)
        print(f'--------------------------------------------------', file=f)

        print(f'--------------------------------------------------', file=f)
        print(f'Simulation parameters:', file=f)
        print(f'Stabilisation parameter: alpha = {alpha}', file=f)
        print(f'Time step: dt = {dt}', file=f)
        print(f'Time interval: [{t_start}, {t_end}]', file=f)
        print(f'Number of steps: {nb_steps}', file=f)
        print(f'Save every: {save_every}', file=f)
        print(f'Evaluation every: {eval_every}', file=f)
        print(f'--------------------------------------------------', file=f)

        print(f'--------------------------------------------------', file=f)
        print(f'Flags:', file=f)
        print(f'Restart: {RESTART}', file=f)
        print(f'Spinodal: {SPINODAL}', file=f)
        print(f'Path: {PATH}', file=f)
        print(f'Evaluation: {EVAL}', file=f)
        print(f'Substrate: {SUBSTRATE}', file=f)
        print(f'--------------------------------------------------', file=f)

    if RESTART:
        # ----------------- RESTART ----------------- #
        filename = f"{PATH}{FILENAME}_{config_name}_{float(t_start):08.1f}.vtk"
        print(f"Restarting from {filename} ...")

        c, config = readVTK(filename)

        if c.ndim > 2:
            nx, ny, nz = c.shape
        else:
            nx, ny = c.shape
            nz = 1

        print(config)

        FILENAME = f"{FILENAME}_{config['type']}"

        dX = config["dX"]
        dx, dy, dz = dX, dX, dX
        Lx, Ly, Lz = nx*dx, ny*dy, nz*dz

        print(f'--------------------------------------------------')
        print(f'Loaded {filename}')
        if config["type"] == "nanowire":
            print(f'Domain size: {Lx} x {Ly} x {Lz}')
            print(f'Computational domain: {nx} x {ny} x {nz}')
            print(f'Mesh size: {dX}')
            print(f'Configuration: nb nanowires = {config["nb_nw"]}')
            print(f'Finite? {FINITE}')
            print(f'Radii: R_1 = {config["r_1"]*dX} and R_2 = {config["r_2"]*dX}')\
                if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]*dX}')
            print(f'Angle: {config["angle"]}')\
                if config["nb_nw"] == 2 else print(f'No orientation specified')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')
        else:
            print(f'Domain size: {Lx} x {Ly} x {Lz}')
            print(f'Mesh size: {dX}')
            print(f'Configuration: {config["type"]}')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')

        # convert to torch tensor
        c = torch.tensor(c, dtype=torch.float32)

        if FINITE:
            # make it finite
            FINITE_GRID_SIZE_Z = int(max(Lz-FINITE_LENGTH, 0.0)/dX)
            FINITE_GRID_SIZE1_Z = FINITE_GRID_SIZE_Z//2
            FINITE_GRID_SIZE2_Z = FINITE_GRID_SIZE_Z - FINITE_GRID_SIZE1_Z
            c[:, :, :FINITE_GRID_SIZE1_Z] = 0.0
            c[:, :, -FINITE_GRID_SIZE2_Z:] = 0.0
            if config["nb_nw"] == 2: # only if Lx == Lz
                FINITE_GRID_SIZE_X = int(max(Lx-FINITE_LENGTH, 0.0)/dX)
                FINITE_GRID_SIZE1_X = FINITE_GRID_SIZE_X//2
                FINITE_GRID_SIZE2_X = FINITE_GRID_SIZE_X - FINITE_GRID_SIZE1_X
                c[:FINITE_GRID_SIZE1_X, :, :] = 0.0
                c[-FINITE_GRID_SIZE2_X:, :, :] = 0.0

        # if t_start < dt:
        #     print("Adding noise to the composition field...")
        #     torch.manual_seed(seed)
        #     random_noise = c_noise * (0.5 - torch.rand(size=c.shape, dtype=torch.float32))
        #     c = c + random_noise

        if SUBSTRATE:
            indic = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], config["r_1"]*dX-1.5, type='planar')
            c = (1-indic) * c
            
        if SBM:
            if PENTAGONAL:
                    H = 1/4 * theta * config["r_1"]*dX / np.tan(np.pi/5) * np.sign(np.cos(theta))
            else:
                H = config["r_1"]*dX*np.cos(theta)
                H = config["r_1"]*dX-1.5
                H = config["r_1"]*dX-2*dX
                H = config["r_1"]*dX
            substrate_position = Ly/2 - H
            #wall = 1.0 - makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], 0, 2*R, type='planar')
            wall = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], substrate_position, type='planar')
            # c = c*wall if not config['substrate'] else c
            plt.figure()
            plt.imshow(c[:, :, nz//2])
            plt.colorbar()
            plt.show()
    else:
        # ----------------- PREPROCESSING ----------------- #
        if not SPINODAL:
            print(f'New simulation...')
            # Set .geo file
            config = loadGeom(sys.argv[1])

            config["seed"] = seed

            # read .geo file
            nx = config["Nx"]
            ny = config["Ny"]
            nz = config["Nz"]

            Lx = config["Lx"]
            Ly = config["Ly"]
            Lz = config["Lz"]

            dX = config["dX"]
            dx, dy, dz = dX, dX, dX

            print(f'--------------------------------------------------')
            print(f'Loaded {sys.argv[1]}')
            print(f'Domain size: {Lx} x {Ly} x {Lz}')
            print(f'Computational domain: {nx} x {ny} x {nz}')
            print(f'Mesh size: {dX}')
            print(f'Configuration: nb nanowires = {config["nb_nw"]}')
            print(f'Radii: R_1 = {config["r_1"]*dX} and R_2 = {config["r_2"]*dX}')\
                if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]*dX}')
            print(f'Angle: {config["angle"]}')\
                if config["nb_nw"] == 2 else print(f'No orientation specified')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')

            FILENAME = f"{FILENAME}_{config['type']}"

            # create a composition field
            print("Preprocessing...")
            # c, coords = makeCompositionFieldParallel([nx, ny, nz], [Lx, Ly, Lz], c_zero, file=sys.argv[1], num_workers=8)

            c = makeCompositionFieldTorch(config, kappa, A, shape=1 if PENTAGONAL else 0, k_mode = k_mode)

            if FINITE and config["type"] == "nanowire":
                # make it finite

                

                FINITE_GRID_SIZE_Z = int(max(Lz-FINITE_LENGTH, 0.0)/dX)
                FINITE_GRID_SIZE_X = int(max(Lx-FINITE_LENGTH, 0.0)/dX)
                FINITE_GRID_SIZE1_Z = FINITE_GRID_SIZE_Z//2
                FINITE_GRID_SIZE2_Z = FINITE_GRID_SIZE_Z - FINITE_GRID_SIZE1_Z
                FINITE_GRID_SIZE1_X = FINITE_GRID_SIZE_X//2
                FINITE_GRID_SIZE2_X = FINITE_GRID_SIZE_X - FINITE_GRID_SIZE1_X
                c[:, :, :FINITE_GRID_SIZE1_Z] = 0.0
                c[:, :, -FINITE_GRID_SIZE2_Z:] = 0.0
                c[:FINITE_GRID_SIZE1_X, :, :] = 0.0 if FINITE_GRID_SIZE_X > 0 else c[:FINITE_GRID_SIZE1_X, :, :]
                c[-FINITE_GRID_SIZE2_X:, :, :] = 0.0 if FINITE_GRID_SIZE_X > 0 else c[-FINITE_GRID_SIZE2_X:, :, :]

            if SUBSTRATE:
                indic = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], config["r_1"]*dX-1.5, type='planar')
                c = (1-indic) * c
            
            config["substrate"] = SBM

            config["L"] = L
            config["E"] = E
            config["T"] = T

            if SBM:
                if PENTAGONAL:
                    H = 1/4 * theta * config["r_1"]*dX / np.tan(np.pi/5) * np.sign(np.cos(theta))
                else:
                    H = config["r_1"]*dX*np.cos(theta)
                    H = config["r_1"]*dX-1.5
                    H = config["r_1"]*dX-3*dX
                    # H = config["r_1"]*dX
                    # H = config["r_1"]*dX
                substrate_position = Ly/2 - H
                #wall = 1.0 - makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], 0, 2*R, type='planar')
                wall = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], A, kappa, substrate_position, type='planar')
                c = c*wall
                # c[wall < 0.5] = 0.0

            # print time 0 frame
            saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=FILENAME)
            saveFrame(wall, t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_wall", desc="wall", eval=False) if SBM else None
            # saveFrame(c*wall, t_start, save_every, config, dt, path=PATH, filename=FILENAME) if SBM else saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=FILENAME)

            # add noise to the composition field
            print("Adding noise to the composition field...")
            torch.manual_seed(seed)
            random_noise = c_noise * (0.5 - torch.rand(c.shape[-1], dtype=torch.float32))
            # #random_noise = torch.randn(size=c.shape, dtype=torch.float32) * c_noise

            # c = c * (1 + random_noise)

            # saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=f'{FILENAME}_noise', eval=False)
        else:
            # spinodal decomposition simulation
            if TEST_NAME == "zhu":
                dX = dx_spinodal
                c_noise = 0.2

                dx, dy, dz = dX, dX, 1.0
                Lx, Ly, Lz = 1024.0, 1024.0, 1.0
                
                nx, ny, nz = int(Lx/dx), int(Ly/dy), int(Lz/dz) # 2D problem
                
                # from Zhu et al.
                torch.manual_seed(seed)

                # 2D problem
                c = 0.5 + c_noise * (0.5 - torch.rand((int(Lx), int(Ly)), dtype=torch.float32)) 

                # avg pool from 1024x1024 to 1024/dx x 1024/dy
                if dx > 1.0:
                    c = torch.nn.functional.avg_pool2d(c[None, None, :, :], 
                                                    int(dx), stride=int(dx)).squeeze()
                elif dx < 1.0:
                    c = torch.nn.functional.interpolate(c[None, None, :, :], 
                                                    scale_factor=1/dx, mode='bilinear').squeeze()
                else:
                    c = c
            elif TEST_NAME == "jokisaari":
                dX = dx_spinodal
                c_noise = 0.01

                dx, dy, dz = dX, dX, 1.0
                Lx, Ly, Lz = 200.0, 200.0, 1.0

                nx, ny, nz = int(Lx/dx), int(Ly/dy), int(Lz/dz) # 2d problem

                # from Jokisaari et al.
                x = torch.linspace(start=0.0, end=Lx, steps=nx)
                y = torch.linspace(start=0.0, end=Ly, steps=ny)
                # z = torch.arange(start=0.0, end=Lz, step=dz)

                #xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
                xx, yy = torch.meshgrid(x, y, indexing='ij')

                c = 0.5 + c_noise * (torch.cos(0.105 * xx) * torch.cos(0.11 * yy) +\
                                    (torch.cos(0.13 * xx) * torch.cos(0.087 * yy))**2 +\
                                    torch.cos(0.025*xx-0.15*yy) * torch.cos(0.07*xx-0.02*yy))
            else:
                dX = dx_spinodal
                c_noise = 1e-3

                dx, dy, dz = dX, dX, 1.0
                # Lx, Ly, Lz = 96.0, 48.0*2.0, 1.0
                Lx, Ly, Lz = 200.0*3, 96.0*2, 1.0
                nx, ny, nz = int(Lx/dx), int(Ly/dy), int(Lz/dz) # 2d problem

                x = torch.linspace(start=0.0, end=Lx, steps=nx)
                y = torch.linspace(start=0.0, end=Ly, steps=ny)

                xx, yy = torch.meshgrid(x, y, indexing='ij')

                R = 3.0

                c = torch.zeros((nx, ny), dtype=torch.float32)
                # c[(xx - Lx/2)**2 + (yy - Ly/2)**2 < R**2] = 1.0
                # c[(yy - Ly/2)**2 < R**2] = 1.0
                eps = np.sqrt(2.0*kappa/A)
                # c = 1/2 * (1.0 + torch.tanh((R-torch.sqrt((yy-Ly/2)**2+(xx-Lx/2)**2))/eps))
                c = 1/2 * (1.0 + torch.tanh((R-torch.sqrt((yy-Ly/2)**2))/eps))
                c[xx>Lx-Lx/5] = 0.0

                torch.manual_seed(seed)

                # c = 0.5 + c_noise * (0.5 - torch.rand((nx, ny), dtype=torch.float32)) 

                # c[xx < 100] = 0.0
                # c[xx > Lx - 100] = 0.0
                
                # c = c + c_noise * (0.5 - torch.rand((nx, ny), dtype=torch.float32))
            if SUBSTRATE:
                #R = 100
                indic = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], 6.0, 2*R, type='circular')
                c = (1-indic) * c
            
            if SBM:
                substrate_position = Ly/2 - R
                wall = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], A, kappa, substrate_position, type='planar')
                # wall = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], substrate_position, type='square')
                # c = c*wall

            config = {"Nx": nx, "Ny": ny, "Nz": nz, 
                      "Lx": nx*dX, "Ly": ny*dX, "Lz": nz*dX, 
                      "dX": dX, "type": "spinodal", "substrate": SBM}

            print(f'New simulation...')
            print(f'--------------------------------------------------')
            print(f'No geometry file specified')
            print(f'Domain size: {Lx} x {Ly} x {Lz}')
            print(f'Computational domain: {nx} x {ny} x {nz}')
            print(f'Mesh size: {dX}')
            print(f'Configuration: {config["type"]}')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')
            
            FILENAME = f"{FILENAME}_{config['type']}"
            
            # c = c*wall if SBM else c

            saveFrame(wall, t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_wall", desc="wall", eval=False) if SBM else None
            saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=FILENAME)
            # saveFrame(c*wall, t_start, save_every, config, dt, path=PATH, filename=FILENAME) if SBM else saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=FILENAME)

    # ----------------- SOLVER ----------------- #
    with torch.no_grad():
        # define fourier modes (using rfft thus only half of the modes on the last axis)
        if c.ndim > 2:
            _kx = torch.fft.fftfreq(c.shape[0], d=dx) * 2.0 * np.pi
            _ky = torch.fft.fftfreq(c.shape[1], d=dy) * 2.0 * np.pi
            _kz = torch.fft.rfftfreq(c.shape[2], d=dz) * 2.0 * np.pi if RFFT else torch.fft.fftfreq(c.shape[2], d=dz) * 2.0 * np.pi

            kx, ky, kz = torch.meshgrid(_kx, _ky, _kz, indexing='ij')

            jkx, jky, jkz = 1j*kx, 1j*ky, 1j*kz

            jk = [jkx, jky, jkz]
            k = [kx, ky, kz]
            k2 = kx*kx + ky*ky + kz*kz
            k4 = k2 * k2

            # Compute the Nyquist frequencies
            kx_max = np.pi / dx
            ky_max = np.pi / dy
            kz_max = np.pi / dz if c.shape[2] == 1 else np.inf

            # Compute the fourier space step size
            dkx = np.abs(_kx[1] - _kx[0])
            dky = np.abs(_ky[1] - _ky[0])
            dkz = np.abs(_kz[1] - _kz[0]) if c.shape[2] > 1 else 0.0

            # Set the filter radius as 2/3 of the smallest Nyquist frequency
            filter_radius = (2/3) * min(kx_max, ky_max, kz_max)
            
            # Set the filter width to ensure smooth transition
            filter_width = 3 * max(dkx, dky, dkz)
            low_pass_filter = 1/2 * (1 + torch.tanh((filter_radius - torch.sqrt(kx**2 + ky**2 + kz**2))/filter_width)) if DIFFUSE else None
            
        else:
            _kx = torch.fft.fftfreq(c.shape[0], d=dx) * 2.0 * np.pi
            _ky = torch.fft.rfftfreq(c.shape[1], d=dy) * 2.0 * np.pi if RFFT\
                else torch.fft.fftfreq(c.shape[1], d=dy) * 2.0 * np.pi

            kx, ky = torch.meshgrid(_kx, _ky, indexing='ij')

            jkx, jky = 1j*kx, 1j*ky

            jk = [jkx, jky]
            k = [kx, ky]
            k2 = kx*kx + ky*ky
            k4 = k2 * k2

            # Calculate the Nyquist frequencies
            kx_max = np.pi / dx
            ky_max = np.pi / dy

            # Set the filter radius as 2/3 of the smallest Nyquist frequency
            filter_radius = (np.sqrt(2)/3) * min(kx_max, ky_max)
            
            # Set the filter width to ensure smooth transition
            filter_width = 3 * max(np.abs(_kx[1] - _kx[0]), np.abs(_ky[1] - _ky[0]))
            low_pass_filter = 1/2 * (1 + torch.tanh((filter_radius - torch.sqrt(kx**2 + ky**2))/filter_width)) if DIFFUSE else None

        for i in range(len(k)):
            k[i] = k[i].to(device=device)
            jk[i] = jk[i].to(device=device)

        k2 = k2.to(device=device)
        k4 = k4.to(device=device)

        if low_pass_filter is not None:
            low_pass_filter = low_pass_filter.to(device=device)
        c_device = c.to(device=device)
        c_old_device = c_device.clone()
        indic_device = indic.to(device=device) if SUBSTRATE else None
        wall_device = wall.to(device=device) if SBM else None

        # preallocate tensors in device before the loop
        torch.set_default_device(device)

        ## --- REAL SPACE --- ##
        g_device = torch.zeros_like(c_device)
        phi_device = torch.zeros_like(c_device)
        temp_device = torch.zeros_like(c_device)
        _mu_device = torch.zeros_like(c_device)
        
        # grad_mu_device
        grad_mu_device = []
        for i in range(c.ndim):
            grad_mu_device.append(torch.zeros_like(c_device))

        # grad_indic_device
        indic_device_fft = compute_fftn(indic_device, filter=low_pass_filter) if SUBSTRATE else None
        if SUBSTRATE:
            grad_indic_device = []
            for i in range(c.ndim):
                grad_indic_device.append(compute_ifftn(jk[i]*indic_device_fft, size=indic_device.size())) if SUBSTRATE else None

        if SUBSTRATE:
            grad_indic_norm_device = torch.zeros_like(c_device)
            for i in range(c.ndim):
                grad_indic_norm_device += grad_indic_device[i]**2
        
            grad_indic_norm_device.sqrt_()

        ## --- SBM FORMALISM --- ##
        if SBM:
            inv_wall_device = 1.0
            # sbm_offset = 1e-3
            # inv_wall_device = 1.0/(wall_device + sbm_offset * (1-wall_device))
            # inv_wall_device = 1.0
            # inv_wall_device = 1.0
            # inv_wall_device = 1.0/(inv_wall_device+sbm_offset)
            
            # Evaluate norm of gradient of psi (wall)
            norm_grad_wall_device = compute_ifftn(jk[0]*compute_fftn(wall_device, filter=low_pass_filter), size=c_device.size())**2 + compute_ifftn(jk[1]*compute_fftn(wall_device, filter=low_pass_filter), size=c_device.size())**2
            
            if c.ndim > 2:
                norm_grad_wall_device.add_(compute_ifftn(jk[2]*compute_fftn(wall_device, filter=low_pass_filter), size=c_device.size())**2)
            
            norm_grad_wall_device.sqrt_()

            # norm_grad_wall_device = torch.abs(wall_device)*torch.abs(1.0-wall_device)

            # saveFrame(norm_grad_wall_device, t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_norm_grad_wall", desc="norm_grad_wall", eval=False)

            # Preallocate tensors for grad x (psi x grad(c))
            grad_wall_x_grad_c_device_fft = []
            for i in range(c.ndim):
                grad_wall_x_grad_c_device_fft.append(torch.zeros_like(k[0], dtype=torch.complex64))
            
            # Preallocate Neumann boundary condition tensor
            neumann_bc_device_fft = torch.zeros_like(k[0], dtype=torch.complex64)

            # Preallocate grad(wall) tensor
            grad_wall_device = []
            grad_wall_device_approx = []
            grad_c_device = []
            for i in range(c.ndim):
                grad_wall_device.append(compute_ifftn(jk[i]*compute_fftn(wall_device, filter=low_pass_filter), size=c_device.size()))
                grad_c_device.append(torch.zeros_like(c_device))

            # evaluate grad(wall) tensor
            # print(torch.gradient(wall_device, dim=2)[0])
            grad_wall_device_approx.append(torch.gradient(wall_device, dim=0)[0])
            grad_wall_device_approx.append(torch.gradient(wall_device, dim=1)[0])
            if c.ndim > 2:
                grad_wall_device_approx.append(torch.gradient(wall_device, dim=2)[0])

            grad_wall_device = grad_wall_device_approx

            # saveFrame(norm_grad_wall_device, t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_norm_grad_wall_approx", desc="norm_grad_wall_approx", eval=False)
            # saveFrame(grad_wall_device[0], t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_grad_wall_x", desc="grad_wall_x", eval=False)
            # saveFrame(grad_wall_device_approx[0], t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_grad_wall_x_approx", desc="grad_wall_x_approx", eval=False)
            # saveFrame(grad_wall_device[1], t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_grad_wall_y", desc="grad_wall_y", eval=False)
            # saveFrame(grad_wall_device_approx[1], t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_grad_wall_y_approx", desc="grad_wall_y_approx", eval=False)
            # if c.ndim > 2:
            #     saveFrame(grad_wall_device[2], t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_grad_wall_z", desc="grad_wall_z", eval=False)
            #     saveFrame(grad_wall_device_approx[2], t_start, save_every, config, dt, path=PATH, filename=f"{FILENAME}_grad_wall_z_approx", desc="grad_wall_z_approx", eval=False)

            # grad1_wall_device = compute_ifftn(jk[0]*compute_fftn(wall_device, filter=low_pass_filter), size=c_device.size())
            # grad2_wall_device = compute_ifftn(jk[1]*compute_fftn(wall_device, filter=low_pass_filter), size=c_device.size())
            # if c.ndim > 2:
            #     grad3_wall_device = compute_ifftn(jk[2]*compute_fftn(wall_device, filter=low_pass_filter), size=c_device.size())
            #     grad_wall_device = [grad1_wall_device, grad2_wall_device, grad3_wall_device]
            # else:
            #     grad_wall_device = [grad1_wall_device, grad2_wall_device]

        ## --- FOURIER SPACE --- ##
        c_device_fft = torch.zeros_like(k[0], dtype=torch.complex64)
        g_device_fft = torch.zeros_like(k[0], dtype=torch.complex64)
        mu_device_fft = torch.zeros_like(k[0], dtype=torch.complex64)

        compute_fftn(c_device, out=c_device_fft, filter=low_pass_filter)

        # flux_device_fft
        flux_device_fft = []
        for i in range(c.ndim):
            flux_device_fft.append(torch.zeros_like(k[0], dtype=torch.complex64))

        if SUBSTRATE:
            beta_device_fft = torch.zeros_like(k[0], dtype=torch.complex64) 
            flux_c_device_fft = torch.zeros_like(k[0], dtype=torch.complex64) 
            bc_device_fft = torch.zeros_like(k[0], dtype=torch.complex64) 
        

    # ----------------- TIME LOOP ----------------- #
    energy_param = [A, kappa, c_zero, dx]
    if EVAL:
        mass, energy, t_arr, mu_arr = [], [], [], []
        if RESTART:
            mass_old = torch.load(f'{PATH}mass_{FILENAME}.pt')
            energy_old = torch.load(f'{PATH}energy_{FILENAME}.pt')
            t_arr_old = torch.load(f'{PATH}time_{FILENAME}.pt')
            mu_arr_old = torch.load(f'{PATH}mu_{FILENAME}.pt')

            mass.extend(mass_old)
            energy.extend(energy_old)
            t_arr.extend(t_arr_old)
            mu_arr.extend(mu_arr_old)

        mass.append(evalTotalMass(c_device, config["dX"]))
        energy.append(evalTotalEnergy(c_device, c_device_fft, jk, energy_param, filter=low_pass_filter))
        t_arr.append(t_start)

    # if EVAL:
    #     mass_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
    #     energy_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
    #     mass_device[0] = evalTotalMass(c_device)
    #     energy_device[0] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz, dx, filter=low_pass_filter, grad_c0_norm_device=grad_indic_norm_device, c0_device=indic_device)
    # idx_eval = 1

    with torch.no_grad():
        FLAG_WARN = True

        t_range = range(int(t_start/dt)+1, int(t_start/dt+nb_steps+1))
        E1 = evalTotalEnergy(c_device, c_device_fft, jk, energy_param, filter=low_pass_filter)
        
        t=int(t_start/dt)+1
        t_elapsed = t_start + dt
        with tqdm(total=nb_steps, desc="Solving...") as pbar:
            pbar.update(t-1)
            while (t_elapsed <= t_end):
                # c_device[wall_device <= 0.5] = 0.0
                # compute g (df/dc)
                compute_g(g_device, c_device, A, c_zero)

                # compute phi (M(c))
                compute_vm(phi_device, c_device, M, a=a)

                # compute mu_hat (F(df/dc) - F(grad^2(kappa/2*c)=F(mu)=mu_hat)
                if SBM:
                    phi_device[wall_device < 0.5] = 0.0
                    phi_device.mul_(wall_device)

                    # compute grad(c)
                    compute_ifftn(jk[0]*c_device_fft, out=grad_c_device[0], size=c_device.size())
                    compute_ifftn(jk[1]*c_device_fft, out=grad_c_device[1], size=c_device.size())
                    if c.ndim > 2:
                        compute_ifftn(jk[2]*c_device_fft, out=grad_c_device[2], size=c_device.size())

                    # saveFrame(phi_device, t, save_every, config, dt, path=PATH, filename=f'{FILENAME}_M', t_elapsed=t_elapsed, eval=False)
                    # saveFrame(g_device, t, save_every, config, dt, path=PATH, filename=f'{FILENAME}_g', t_elapsed=t_elapsed, eval=False)

                    # compute_fftn(g_device, out=g_device_fft, filter=low_pass_filter)
                    # compute_mu_fft_SBM(mu_device_fft, c_device, c_device_fft, g_device_fft, wall_device, inv_wall_device, grad_wall_x_grad_c_device_fft, neumann_bc_device_fft, temp_device, norm_grad_wall_device, [A, kappa, theta], jk, low_pass_filter)

                    # compute_mu_fft_SBM3(mu_device_fft, c_device, c_device_fft, g_device, wall_device, inv_wall_device, norm_grad_wall_device, [A, kappa, theta], jk, low_pass_filter)

                    # compute_ifftn(mu_device_fft, out=temp_device, size=c_device.size())
                    # saveFrame(temp_device, t, save_every, config, dt, path=PATH, filename=f'{FILENAME}_mu', t_elapsed=t_elapsed, eval=False)
                    compute_mu_fft_SBM2(mu_device_fft, c_device, c_device_fft, grad_c_device, g_device, wall_device, inv_wall_device, norm_grad_wall_device, [A, kappa, theta], jk, filter=low_pass_filter)
                elif SUBSTRATE:
                    phi_device.mul_(1.0-indic_device)
                    compute_fftn(g_device, out=g_device_fft, filter=low_pass_filter)
                    compute_mu_fft_bis(mu_device_fft, c_device, c_device_fft, flux_c_device_fft, bc_device_fft, g_device_fft,
                                    grad_indic_device, grad_indic_norm_device, low_pass_filter,
                                    kappa, k, k2, theta)
                    # compute_fftn(36*A*c_device*c_device*(1-c_device-indic_device)*indic_device, out=beta_device_fft, filter=low_pass_filter)
                    # compute_fftn(2*A*c_device*indic_device*(1-indic_device)*(1-2*indic_device), out=beta_device_fft, filter=low_pass_filter)
                    # compute_fftn(4*A*c_device*(1-c_device-indic_device)*indic_device, out=beta_device_fft, filter=low_pass_filter)
                    # mu_device_fft.add_(-beta_device_fft)
                else:
                    compute_fftn(g_device, out=g_device_fft, filter=low_pass_filter)
                    compute_mu_fft(mu_device_fft, c_device_fft, g_device_fft, kappa, k2)

                # compute grad_mu_hat (F(grad(mu)) = jk * mu_hat )
                #compute_grad_mu([grad_mu1_device, grad_mu2_device, grad_mu3_device], mu_device_fft, jk)

                compute_grad_mu(grad_mu_device, mu_device_fft, k)

                # grad_mu_device[0][wall_device <= 0.5] = 0.0
                # grad_mu_device[1][wall_device <= 0.5] = 0.0
                # if c.ndim > 2:
                #     grad_mu_device[2][wall_device <= 0.5] = 0.0

                # compute composition flux (phi * F-1(jk * r))
                #compute_flux_fft([flux1_device_fft, flux2_device_fft, flux3_device_fft], [grad_mu1_device, grad_mu2_device, grad_mu3_device], phi_device, low_pass_filter)
                compute_flux_fft(flux_device_fft, grad_mu_device, phi_device, low_pass_filter)

                # update c (dc/dt = jk * F(q) --> semi-implicit scheme)
                #compute_euler_timestep(c_device_fft, [flux1_device_fft, flux2_device_fft, flux3_device_fft], jk, k4, [kappa, alpha, dt])
                compute_euler_timestep(c_device_fft, flux_device_fft, jk, k4, [kappa, alpha, dt]) if not SBM else compute_euler_timestep(c_device_fft, flux_device_fft, jk, k4, [kappa, alpha, dt], inv_wall_device, filter=low_pass_filter)

                # bring back to real space
                compute_ifftn(c_device_fft, out=c_device)

                # c_device.mul_(wall_device)

                # compute_fftn(c_device, out=c_device_fft, filter=low_pass_filter)

                # print once when condition is met
                if FLAG_WARN and torch.max(c_device).item() < 0.5:
                    print(f'{t_elapsed}: Warning! Order parameter c stricly below critical value.')
                    FLAG_WARN = False

                # save frame
                saveFrame(c_device, t, save_every, config, dt, path=PATH, filename=FILENAME, t_elapsed=t_elapsed)
                # saveFrame(c_device, t, save_every, config, dt, path=PATH, filename=FILENAME, t_elapsed=t_elapsed) if not SBM else saveFrame(c_device*wall_device, t, save_every, config, dt, path=PATH, filename=FILENAME, t_elapsed=t_elapsed)

                if ADAPTIVE:
                    E2 = evalTotalEnergy(c_device, jk, energy_param, filter=low_pass_filter)
                    dEdt_sq = np.abs((E2 - E1)/dt)**2
                    dt = max(dt_min, dt_max/(np.sqrt(1+beta*dEdt_sq)))
                    E1 = E2
                
                # update timestep
                nb_steps_left = int((t_end - t_elapsed)/dt)
                pbar.total = nb_steps_left + t
                # pbar.refresh()
                pbar.update(1)
                t_elapsed = t_elapsed + dt
                t = t + 1

                # evaluate mass and energy
                if EVAL: #and t % eval_every == 0:
                    mass.append(evalTotalMass(c_device, config["dX"]))
                    energy.append(evalTotalEnergy(c_device, c_device_fft, jk, energy_param, filter=low_pass_filter))
                    compute_ifftn(mu_device_fft, out=_mu_device, size=c_device.size())
                    mu_max = torch.max(_mu_device)
                    mu_min = torch.min(_mu_device)
                    mu_arr.append(mu_max-mu_min)
                    
                    # saveFrame(_mu_device, t, save_every, config, dt, path=PATH, filename=f'{FILENAME}_mu', desc='mu_field', t_elapsed=t_elapsed, eval=False)

                    t_arr.append(t_elapsed)
                    #mu_max.append(torch.max(compute_ifftn(mu_device_fft, out=None, size=c_device.size()).abs()))
                
                # evaluate numerical cos(theta)
                #if SBM:
                    #saveFrame(evalCosThetaB(c_device, c_device_fft, grad_wall_device, norm_grad_wall_device, jk, [kappa, A], filter=low_pass_filter), t, save_every, config, dt, path=PATH, filename=f'{FILENAME}_cos', t_elapsed=t_elapsed, eval=False)

            # if EVAL and t % eval_every == 0:
            #     mass_device[idx_eval] = evalTotalMass(c_device)
            #     energy_device[idx_eval] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz, dx)
            #     idx_eval += 1

    # ----------------- POST-PROCESSING ----------------- #
    if EVAL:
        torch.save(torch.tensor(mass, device='cpu'), f"{PATH}mass_{FILENAME}.pt")
        torch.save(torch.tensor(energy, device='cpu'), f"{PATH}energy_{FILENAME}.pt")
        torch.save(torch.tensor(t_arr, device='cpu'), f"{PATH}time_{FILENAME}.pt")
        mu_arr.append(mu_arr[-1])
        mu_tensor = torch.tensor(mu_arr, device='cpu')
        mu_tensor = mu_tensor/mu_tensor[0]
        torch.save(mu_tensor, f"{PATH}mu_{FILENAME}.pt")

    # clear memory
    del grad_mu_device
    del k
    del jk
    del k2, k4
    del c_device_fft, g_device_fft, 
    del c_device, g_device, phi_device

