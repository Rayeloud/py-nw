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
    if "-substrate" in sys.argv:
        option = sys.argv.index("-substrate")
        SUBSTRATE = True
        substrate_dim = float(sys.argv[option+1])
    if "-dt" in sys.argv:
        option = sys.argv.index("-dt")
        dt = float(sys.argv[option+1])
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

    c = None

    eval_every = dt

    if isinstance(save_every, list):
        for i, freq in enumerate(save_every):
            save_every[i] = int(freq/dt)
    else:
        save_every = int(save_every/dt)
    eval_every = int(eval_every/dt)

    nb_steps = int((t_end-t_start)/dt)

    theta = theta * np.pi/180

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
        filename = f"{PATH}{FILENAME}_{int(t_start/dt):06}.vtk"
        print(f"Restarting from {filename} ...")

        c, config = readVTK(filename)

        if c.ndim > 2:
            nx, ny, nz = c.shape
        else:
            nx, ny = c.shape
            nz = 1

        print(config)

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
            FINITE_GRID_SIZE_X = int(max(Lx-FINITE_LENGTH, 0.0)/dX)
            FINITE_GRID_SIZE1_Z = FINITE_GRID_SIZE_Z//2
            FINITE_GRID_SIZE2_Z = FINITE_GRID_SIZE_Z - FINITE_GRID_SIZE1_Z
            FINITE_GRID_SIZE1_X = FINITE_GRID_SIZE_X//2
            FINITE_GRID_SIZE2_X = FINITE_GRID_SIZE_X - FINITE_GRID_SIZE1_X
            c[:, :, :FINITE_GRID_SIZE1_Z] = 0.0
            c[:, :, -FINITE_GRID_SIZE2_Z:] = 0.0
            c[:FINITE_GRID_SIZE1_X, :, :] = 0.0
            c[-FINITE_GRID_SIZE2_X:, :, :] = 0.0


        if t_start == 0:
            print("Adding noise to the composition field...")
            torch.manual_seed(seed)
            random_noise = c_noise * (0.5 - torch.rand(size=c.shape, dtype=torch.float32))
            c = c + random_noise
        
    else:
        # ----------------- PREPROCESSING ----------------- #
        if not SPINODAL:
            print(f'New simulation...')
            # Set .geo file
            config = loadGeom(sys.argv[1])

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
            c, coords = makeCompositionFieldParallel([nx, ny, nz], [Lx, Ly, Lz], c_zero, file=sys.argv[1], num_workers=8)

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

            # print time 0 frame
            saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=FILENAME)

            # add noise to the composition field
            print("Adding noise to the composition field...")
            torch.manual_seed(seed)
            random_noise = c_noise * (0.5 - torch.rand(size=c.shape, dtype=torch.float32))
            #random_noise = torch.randn(size=c.shape, dtype=torch.float32) * c_noise

            c = c + random_noise
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
                c = 0.0 + c_noise * (0.5 - torch.rand((int(Lx), int(Ly)), dtype=torch.float32)) 

                # avg pool from 1024x1024 to 1024/dx x 1024/dy
                if dx > 1.0:
                    c = torch.nn.functional.avg_pool2d(c[None, None, :, :], 
                                                    int(dx), stride=int(dx)).squeeze()
                elif dx < 1.0:
                    c = torch.nn.functional.interpolate(c[None, None, :, :], 
                                                    scale_factor=1/dx, mode='bilinear').squeeze()
                else:
                    c = c
            else:
                dX = dx_spinodal
                c_noise = 0.01

                dx, dy, dz = dX, dX, 1.0
                Lx, Ly, Lz = 200.0, 200.0, 1.0

                nx, ny, nz = int(Lx/dx), int(Ly/dy), int(Lz/dz) # 2d problem

                # from Jokisaari et al.
                x = torch.arange(start=0.0, end=Lx, step=dx)
                y = torch.arange(start=0.0, end=Ly, step=dy)
                # z = torch.arange(start=0.0, end=Lz, step=dz)

                #xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
                xx, yy = torch.meshgrid(x, y, indexing='ij')

                c = 0.5 + c_noise * (torch.cos(0.105 * xx) * torch.cos(0.11 * yy) +\
                                    (torch.cos(0.13 * xx) * torch.cos(0.087 * yy))**2 +\
                                    torch.cos(0.025*xx-0.15*yy) * torch.cos(0.07*xx-0.02*yy))
            
            if SUBSTRATE:
                indic = makeIndicatorFunction([nx, ny, nz], [Lx, Ly, Lz], 400, type='circular')
                c = (1-indic) * c

            config = {"Nx": nx, "Ny": ny, "Nz": nz, 
                      "Lx": nx*dX, "Ly": ny*dX, "Lz": nz*dX, 
                      "dX": dX, "type": "spinodal"}

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
            
            saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=FILENAME)

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
            low_pass_filter = 1/2 * (1 + torch.tanh((filter_radius - torch.sqrt(kx**2 + ky**2 + kz**2))/filter_width))

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
            filter_radius = (2/3) * min(kx_max, ky_max)
            
            # Set the filter width to ensure smooth transition
            filter_width = 3 * max(np.abs(_kx[1] - _kx[0]), np.abs(_ky[1] - _ky[0]))
            low_pass_filter = 1/2 * (1 + torch.tanh((filter_radius - torch.sqrt(kx**2 + ky**2))/filter_width))

        for i in range(len(k)):
            k[i] = k[i].to(device=device)
            jk[i] = jk[i].to(device=device)

        k2 = k2.to(device=device)
        k4 = k4.to(device=device)

        if low_pass_filter is not None:
            low_pass_filter = low_pass_filter.to(device=device)
        c_device = c.to(device=device)
        indic_device = indic.to(device=device) if SUBSTRATE else None

        # preallocate tensors in device before the loop
        torch.set_default_device(device)

        ## --- REAL SPACE --- ##
        g_device = torch.zeros_like(c_device)
        phi_device = torch.zeros_like(c_device)
        
        # grad_mu_device
        grad_mu_device = []
        for i in range(c.ndim):
            grad_mu_device.append(torch.zeros_like(c_device))

        # grad_indic_device
        indic_device_fft = compute_fftn(indic_device, filter=low_pass_filter) if SUBSTRATE else None
        if SUBSTRATE:
            grad_indic_device = []
            for i in range(c.ndim):
                grad_indic_device.append(compute_ifftn(jk[i]*indic_device_fft, filter=low_pass_filter, size=indic_device.size())) if SUBSTRATE else None

        if SUBSTRATE:
            grad_indic_norm_device = torch.zeros_like(c_device)
            for i in range(c.ndim):
                grad_indic_norm_device += grad_indic_device[i]**2
        
            grad_indic_norm_device.sqrt_()

        ## --- FOURIER SPACE --- ##
        c_device_fft = torch.zeros_like(k[0], dtype=torch.complex64)
        g_device_fft = torch.zeros_like(k[0], dtype=torch.complex64)
        mu_device_fft = torch.zeros_like(k[0], dtype=torch.complex64)
        # flux_device_fft
        flux_device_fft = []
        for i in range(c.ndim):
            flux_device_fft.append(torch.zeros_like(k[0], dtype=torch.complex64))

        if SUBSTRATE:
            beta_device_fft = torch.zeros_like(k[0], dtype=torch.complex64) 
            flux_c_device_fft = torch.zeros_like(k[0], dtype=torch.complex64) 
            bc_device_fft = torch.zeros_like(k[0], dtype=torch.complex64) 
        

    # ----------------- TIME LOOP ----------------- #
    if EVAL:
        mass = []
        energy = []
        energy_param = [A, kappa, c_zero, dx]

        mass.append(evalTotalMass(c_device, dx))
        energy.append(evalTotalEnergy(c_device, jk, energy_param, filter=low_pass_filter))

    # if EVAL:
    #     mass_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
    #     energy_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
    #     mass_device[0] = evalTotalMass(c_device)
    #     energy_device[0] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz, dx, filter=low_pass_filter, grad_c0_norm_device=grad_indic_norm_device, c0_device=indic_device)

    # idx_eval = 1

  

    with torch.no_grad():
        FLAG_WARN = True
        t_range = range(int(t_start/dt)+1, int(t_start/dt+nb_steps+1))
        for t in tqdm(t_range, desc="Solving..."):
            # compute g (df/dc)
            compute_g(g_device, c_device, A, c_zero)

            # compute phi (M(c))
            compute_vm(phi_device, c_device, M, a=a)
            
            phi_device.mul_(1.0-indic_device) if SUBSTRATE else None
            
            compute_fftn(g_device, out=g_device_fft, filter=low_pass_filter)
            compute_fftn(c_device, out=c_device_fft, filter=low_pass_filter)
            
            # compute mu_hat (F(df/dc) - F(grad^2(kappa/2*c)=F(mu)=mu_hat)
            if not SUBSTRATE:
                compute_mu_fft(mu_device_fft, c_device_fft, g_device_fft, kappa, k2)
            else:
                compute_mu_fft_bis(mu_device_fft, c_device, c_device_fft, flux_c_device_fft, bc_device_fft, g_device_fft,
                                   grad_indic_device, grad_indic_norm_device, low_pass_filter,
                                   kappa, k, k2, theta)
                # compute_fftn(36*A*c_device*c_device*(1-c_device-indic_device)*indic_device, out=beta_device_fft, filter=low_pass_filter)
                # compute_fftn(4*A*c_device*(1-c_device-indic_device)*indic_device, out=beta_device_fft, filter=low_pass_filter)
                # mu_device_fft.add_(-beta_device_fft)

            # compute grad_mu_hat (F(grad(mu)) = jk * mu_hat )
            #compute_grad_mu([grad_mu1_device, grad_mu2_device, grad_mu3_device], mu_device_fft, jk)
            compute_grad_mu(grad_mu_device, mu_device_fft, k)

            # compute composition flux (phi * F-1(jk * r))
            #compute_flux_fft([flux1_device_fft, flux2_device_fft, flux3_device_fft], [grad_mu1_device, grad_mu2_device, grad_mu3_device], phi_device, low_pass_filter)
            compute_flux_fft(flux_device_fft, grad_mu_device, phi_device, low_pass_filter)

            # update c (dc/dt = jk * F(q) --> semi-implicit scheme)
            #compute_euler_timestep(c_device_fft, [flux1_device_fft, flux2_device_fft, flux3_device_fft], jk, k4, [kappa, alpha, dt])
            compute_euler_timestep(c_device_fft, flux_device_fft, k, k4, [kappa, alpha, dt])

            # bring back to real space
            compute_ifftn(c_device_fft, out=c_device)

            # print once when condition is met
            if FLAG_WARN:
                if torch.max(c_device).item() < 0.5:
                    print(f'{t*dt}: Warning! Order parameter c stricly below critical value.')
                    FLAG_WARN = False

            saveFrame(c_device, t, save_every, config, dt, path=PATH, filename=FILENAME)
            if EVAL and t % eval_every == 0:
                mass.append(evalTotalMass(c_device, dx))
                energy.append(evalTotalEnergy(c_device, jk, energy_param, filter=low_pass_filter))

            # if EVAL and t % eval_every == 0:
            #     mass_device[idx_eval] = evalTotalMass(c_device)
            #     energy_device[idx_eval] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz, dx)
            #     idx_eval += 1

    # ----------------- POST-PROCESSING ----------------- #
    if EVAL:
        torch.save(torch.tensor(mass, device='cpu'), f"{PATH}mass_{FILENAME}.pt")
        torch.save(torch.tensor(energy, device='cpu'), f"{PATH}energy_{FILENAME}.pt")
    
    # clear memory
    del grad_mu_device
        #del flux_c_device_fft[i]
    del k
    del jk
    del k2, k4
    del c_device_fft, g_device_fft, 
    del c_device, g_device, phi_device

