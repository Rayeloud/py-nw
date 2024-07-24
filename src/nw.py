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

# time params
dt = 2.0
t_start = 0
t_end = 1000
save_every = [10, 50, 100, 250, 500, 1000]
eval_every = 100

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
SUBSTRATE = False
substrate_dim = 0.0

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

    c = None

    if isinstance(save_every, list):
        for i, freq in enumerate(save_every):
            save_every[i] = int(freq/dt)
    else:
        save_every = int(save_every/dt)
    eval_every = int(eval_every/dt)

    nb_steps = int((t_end-t_start)/dt)

    if RESTART:
        # ----------------- RESTART ----------------- #
        filename = f"{PATH}{FILENAME}_{int(t_start/dt):06}.vtk"
        print(f"Restarting from {filename} ...")

        c, config = readVTK(filename)

        nx, ny, nz = c.shape

        dX = config["dX"]
        dx, dy, dz = dX, dX, dX
        Lx, Ly, Lz = nx*dx, ny*dy, nz*dz

        print(f'--------------------------------------------------')
        print(f'Loaded {filename}')
        print(f'Domain size: {Lx} x {Ly} x {Lz}')
        print(f'Mesh size: {dX}')
        print(f'Configuration: nb nanowires = {config["nb_nw"]}')
        print(f'Radii: R_1 = {config["r_1"]} and R_2 = {config["r_2"]}')\
            if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]}')
        print(f'Angle: {config["angle"]}')\
            if config["nb_nw"] == 2 else print(f'No orientation specified')
        print(f'\nSolving on Device: {device}')
        print(f'--------------------------------------------------')

        # convert to torch tensor
        c = torch.tensor(c, dtype=torch.float32)

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
            print(f'Mesh size: {dX}')
            print(f'Configuration: nb nanowires = {config["nb_nw"]}')
            print(f'Radii: R_1 = {config["r_1"]} and R_2 = {config["r_2"]}')\
                if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]}')
            print(f'Angle: {config["angle"]}')\
                if config["nb_nw"] == 2 else print(f'No orientation specified')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')

            theta = 72
            theta = theta * np.pi/180
            # create a composition field
            print("Preprocessing...")
            c, coords = makeCompositionFieldParallel((nx, ny, nz), (Lx, Ly, Lz), c_zero, file=sys.argv[1], num_workers=8)

            if SUBSTRATE:
                indic = makeIndicatorFunction(Lx, Ly, Lz, dx, dy, dz, config["r_1"]*dX-1.5, type='planar')

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
            dX = 0.5
            dx, dy, dz = dX, dX, 1.0
            #Lx, Ly, Lz = 256, 128, 1.0
            Lx, Ly, Lz = 384/2, 48, 1.0
            nx, ny, nz = int(Lx//dx), int(Ly//dy), int(Lz//dz) # 2d problem
            c_noise = 1e-3
            torch.manual_seed(seed)
            # c is initialized as rectangular shape with random noise in the center of the domain

            c = torch.zeros((nx, ny, nz), dtype=torch.float32)
            xx, yy, zz = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz))
            # 0 outside 1 inside circle
            c = (((xx-nx//2)**2 + (yy-ny//2)**2 < 12**2).float())
            c += c_noise * (0.5 - torch.rand((nx, ny, nz), dtype=torch.float32))
            #c = 0.5 + c_noise * (0.5 - torch.rand((nx, ny, nz), dtype=torch.float32))
            
            if SUBSTRATE:
                indic = makeIndicatorFunction(Lx, Ly, Lz, dx, dy, dz, (12*3/4)*0.5, type='planar')
                # idx = 307#int((Ly-Ly//2-substrate_dim)//dx)
                #print(idx)
                #print(torch.where(indic < 1.0))
                #c[:, 304-1, :] = c[:, 304, :]
                #c[:, 0, :] = c[:, -1, :]
                # c = c - 0.5
                # c.mul_(indic)
                # c = c + 0.5
                print(indic.shape, flush=True)
                print(c.shape, flush=True)
                c = (1-indic) * c
                print(c.shape, flush=True)

            config = {"Nx": nx, "Ny": ny, "Nz": nz, 
                      "Lx": nx*dX, "Ly": ny*dX, "Lz": nz*dX, 
                      "dX": dX, "type": "spinodal"}

            print(f'New simulation...')
            print(f'--------------------------------------------------')
            print(f'No geometry file specified')
            print(f'Domain size: {Lx} x {Ly} x {Lz}')
            print(f'Mesh size: {dX}')
            print(f'Configuration: spinodal decomposition')
            print(f'\nSolving on Device: {device}')
            print(f'--------------------------------------------------')
            theta = 72
            FILENAME = f'{FILENAME}_spinodal_{theta}_bis'
            theta = theta * np.pi/180
            saveFrame(c, t_start, save_every, config, dt, path=PATH, filename=FILENAME)

    # ----------------- SOLVER ----------------- #
    with torch.no_grad():
        # define fourier modes
        _kx = torch.fft.fftfreq(c.shape[0], d=dx) * 2.0 * np.pi
        _ky = torch.fft.fftfreq(c.shape[1], d=dy) * 2.0 * np.pi
        _kz = torch.fft.fftfreq(c.shape[2], d=dz) * 2.0 * np.pi

        kx, ky, kz = torch.meshgrid(_kx, _ky, _kz, indexing='ij')

        k2 = kx**2 + ky**2 + kz**2
        k4 = k2*k2

        # bring to device
        kx = kx.to(device=device)
        ky = ky.to(device=device)
        kz = kz.to(device=device)

        k2 = k2.to(device=device)
        k4 = k4.to(device=device)
    
        c_device = c.to(device=device)
        indic_device = indic.to(device=device) if SUBSTRATE else None

        # preallocate tensors in device before the loop
        torch.set_default_device(device)
        # real space
        g_device = torch.zeros_like(c_device)
        phi_device = torch.zeros_like(c_device)
        
        # grad_mu_device
        grad_mu1_device = torch.zeros_like(c_device)
        grad_mu2_device = torch.zeros_like(c_device)
        grad_mu3_device = torch.zeros_like(c_device)
        # grad_indic_device
        indic_device_fft = torch.fft.fftn(indic_device, dim=(0, 1, 2)) if SUBSTRATE else None
        grad_indic1_device = torch.zeros_like(c_device) if SUBSTRATE else None
        grad_indic2_device = torch.zeros_like(c_device) if SUBSTRATE else None
        grad_indic3_device = torch.zeros_like(c_device) if SUBSTRATE else None

        grad_indic1_device.copy_(torch.fft.ifftn(1j*kx*indic_device_fft, dim=(0, 1, 2))) if SUBSTRATE else None
        grad_indic2_device.copy_(torch.fft.ifftn(1j*ky*indic_device_fft, dim=(0, 1, 2))) if SUBSTRATE else None
        grad_indic3_device.copy_(torch.fft.ifftn(1j*kz*indic_device_fft, dim=(0, 1, 2))) if SUBSTRATE else None
        grad_indic_norm_device = torch.sqrt(grad_indic1_device**2 + grad_indic2_device**2 + grad_indic3_device**2) if SUBSTRATE else None

        # fourier space
        c_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        g_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        mu_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        source_term_device_fft = torch.zeros_like(c_device, dtype=torch.complex64) if SUBSTRATE else None
        # flux_device_fft
        flux1_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        flux2_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        flux3_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)


        flux_c_device_fft = torch.zeros_like(c_device, dtype=torch.complex64) if SUBSTRATE else None
        bc_device_fft = torch.zeros_like(c_device, dtype=torch.complex64) if SUBSTRATE else None

    # ----------------- TIME LOOP ----------------- #

    if EVAL:
        mass_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
        energy_device = torch.zeros(nb_steps//eval_every+1, dtype=torch.float32)
        mass_device[0] = evalTotalMass(c_device)
        energy_device[0] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz, dx)

    idx_eval = 1
    with torch.no_grad():
        for t in tqdm(range(int(t_start/dt)+1, t_start+nb_steps+1), desc="Solving..."):
            # compute g (df/dc)
            compute_g(g_device, c_device, A)

            # compute phi (M(c))
            compute_vm(phi_device, c_device, M)
            # indic = 1 in substrate region and 0 elsewhere ; phi_device = 0*phi_device in substrate region and phi_device elsewhere
            # express the following line as a single line of code
            phi_device.mul_(1.0-indic_device) if SUBSTRATE else None

            # phi_device.mul_(indic_device) if SUBSTRATE else None
            
            torch.fft.rfftn(g_device, out=g_device_fft, dim=(0, 1, 2))
            torch.fft.rfftn(c_device, out=c_device_fft, dim=(0, 1, 2))
            torch.fft.rfftn(4*A*c_device*(1-c_device-indic_device)*indic_device, out=source_term_device_fft, dim=(0, 1, 2)) if SUBSTRATE else None

            # compute mu_hat (F(df/dc) - F(grad^2(kappa/2*c)=F(mu)=mu_hat)
            compute_mu_fft(mu_device_fft, c_device_fft, g_device_fft, kappa, k2) if not SUBSTRATE else \
                compute_mu_fft_bis(mu_device_fft, c_device, c_device_fft, flux_c_device_fft, bc_device_fft, g_device_fft, (grad_indic1_device, grad_indic2_device, grad_indic3_device), grad_indic_norm_device,kappa, kx, ky, kz, k2, theta)
            mu_device_fft.add_(-source_term_device_fft) if SUBSTRATE else None

            # compute grad_mu_hat (F(grad(mu)) = jk * mu_hat )
            compute_grad_mu((grad_mu1_device, grad_mu2_device, grad_mu3_device), mu_device_fft, kx, ky, kz)

            # compute composition flux (phi * F-1(jk * r))
            compute_flux_fft((flux1_device_fft, flux2_device_fft, flux3_device_fft), grad_mu1_device, grad_mu2_device, grad_mu3_device, phi_device)

            # update c (dc/dt = jk * F(q) --> semi-implicit scheme)
            compute_euler_timestep(c_device_fft, flux1_device_fft, flux2_device_fft, flux3_device_fft, kx, ky, kz, k4, kappa, alpha, dt)

            # bring back to real space
            c_device.copy_(torch.fft.ifftn(c_device_fft, dim=(0, 1, 2), out=c_device_fft))

            if torch.max(c_device).item() < 0.5:
                print(f'Warning! Order parameter c stricly below critical value.')

            saveFrame(c_device, t, save_every, config, dt, path=PATH, filename=FILENAME)
            if EVAL and t % eval_every == 0:
                mass_device[idx_eval] = evalTotalMass(c_device)
                energy_device[idx_eval] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz, dx)
                idx_eval += 1

    # ----------------- POST-PROCESSING ----------------- #
    if EVAL:
        torch.save(mass_device.cpu(), f"{PATH}mass_{FILENAME}.pt")
        torch.save(energy_device.cpu(), f"{PATH}energy_{FILENAME}.pt")
    
    # clear memory
    del kx, ky, kz, k2, k4
    del c_device, g_device, phi_device, grad_mu1_device, grad_mu2_device, grad_mu3_device
    del c_device_fft, g_device_fft, mu_device_fft
    del flux1_device_fft, flux2_device_fft, flux3_device_fft
