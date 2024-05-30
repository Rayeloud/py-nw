"""
nw.py

This script sets up and runs a simulation for a phase-field model. 

The model parameters include composition parameters (initial composition, 
noise in composition, and seed for random number generator), 
time parameters (time step, end time, and frequency of output), 
and solver parameters (mobility, height of the double well potential, 
and gradient energy coefficient).

The script imports necessary modules from `solver` and `processing`.
"""

from solver import *
from processing import *


# composition params
c_zero = 1.0
c_noise = 1e-3
seed = 3009

# time params
dt = 1.0
t_end = 1000
freq = 100 #[0, 100//dt, 2000//dt, 10000//dt, 30000//dt]

# solver params
M = 1.0
A = 1.0
kappa = 1.0
alpha = 0.5

name = "time"

if __name__ == '__main__':
    # ----------------- DEVICE ----------------- #

    Device = MyDevice()
    Device_name = Device.device_name

    # ----------------- PREPROCESSING ----------------- #

    # get arguments
    if len(sys.argv) < 2:
        print("Usage: python nw.py <file.geo> [-restart <time>] [-spinodal] [-path <path>]")
        sys.exit(1)

    # find flag for post-processing or restart of sim (-pproc or -restart) in sys.argv
    RESTART = False
    SPINODAL = False
    PATH = "../output/"
    EVAL = False
    SAVE = True
    t_start = 0

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
    if "-no-save" in sys.argv:
        SAVE = False

    c = None

    nb_steps = int((t_end-t_start)/dt)

    if RESTART:
        # ----------------- RESTART ----------------- #
        filename = f"{PATH}time_{t_start:06}.vtk"
        print(f"Restarting from {filename} ...")

        c, config = readVTK(filename)

        nx, ny, nz = c.shape

        dX = config["dX"]
        dx, dy, dz = dX, dX, dX

        print(f'--------------------------------------------------')
        print(f'Loaded {filename}')
        print(f'Domain size: {nx} x {ny} x {nz}')
        print(f'Configuration: nb nanowires = {config["nb_nw"]}')
        print(f'Radii: R_1 = {config["r_1"]} and R_2 = {config["r_2"]}')\
            if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]}')
        print(f'Angle: {config["angle"]}')\
            if config["nb_nw"] == 2 else print(f'No orientation specified')
        print(f'\nSolving on Device: {Device_name}')
        print(f'--------------------------------------------------')

        # convert to torch tensor
        c = torch.tensor(c, dtype=torch.float32)

    else:
        # ----------------- PREPROCESSING ----------------- #
        if not SPINODAL:
            print(f'New simulation...')
            # Set .geo file
            tread = time.time()
            config = loadGeom(sys.argv[1])

            print(f'--------------------------------------------------')
            print(f'Loaded {sys.argv[1]}')
            print(f'Domain size: {config["Nx"]} x {config["Ny"]} x {config["Nz"]}')
            print(f'Configuration: nb nanowires = {config["nb_nw"]}')
            print(f'Radii: R_1 = {config["r_1"]} and R_2 = {config["r_2"]}')\
                if config["nb_nw"] == 2 else print(f'Radius: R = {config["r_1"]}')
            print(f'Angle: {config["angle"]}')\
                if config["nb_nw"] == 2 else print(f'No orientation specified')
            print(f'\nSolving on Device: {Device_name}')
            print(f'--------------------------------------------------')

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

            # get the sorted composition field, sorted indices and sorted node tags
            #c, coords = sortByCoords(c, coords)

            # add noise to the composition field
            print("Adding noise to the composition field...")
            torch.manual_seed(seed)
            random_noise = c_noise * (0.5 - torch.rand(size=c.shape))

            c = c + random_noise
        else:
            # spinodal decomposition simulation
            dX = 1.0
            dx, dy, dz = dX, dX, dX
            nx, ny, nz = int(1024//dX), int(1024//dX), 1
            torch.manual_seed(seed)
            c = 0.5 + 0.05 * (0.5 - torch.rand((nx, ny, nz), dtype=torch.float32))

            config = {"Nx": nx, "Ny": ny, "Nz": nz, 
                      "Lx": nx*dX, "Ly": ny*dX, "Lz": nz*dX, 
                      "dX": dX, "type": "spinodal"}

            print(f'New simulation...')
            print(f'--------------------------------------------------')
            print(f'No geometry file specified')
            print(f'Domain size: {nx} x {ny} x {nz}')
            print(f'Configuration: spinodal decomposition')
            print(f'\nSolving on Device: {Device_name}')
            print(f'--------------------------------------------------')

    # ----------------- SOLVER ----------------- #
    with torch.no_grad():
        # define fourier modes
        kx = torch.fft.fftfreq(c.shape[0], d=dx) * 2.0 * np.pi
        ky = torch.fft.fftfreq(c.shape[1], d=dy) * 2.0 * np.pi
        kz = torch.fft.fftfreq(c.shape[2], d=dz) * 2.0 * np.pi

        kx_mesh, ky_mesh, kz_mesh = torch.meshgrid(kx, ky, kz, indexing='ij')

        k2 = kx_mesh**2 + ky_mesh**2 + kz_mesh**2
        k4 = k2*k2

        # bring to device
        kx_mesh = kx_mesh.to(device=Device_name)
        ky_mesh = ky_mesh.to(device=Device_name)
        kz_mesh = kz_mesh.to(device=Device_name)

        k2 = k2.to(device=Device_name)
        k4 = k4.to(device=Device_name)

        c_device = c.to(device=Device_name)
        zeros_device = torch.zeros_like(c_device, device=Device_name)

        # preallocate tensors in device before the loop
        torch.set_default_device(Device_name)

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
        y1_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        y2_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        y3_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        q1_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        q2_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        q3_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)

    if not RESTART and SAVE:
        writeVTK(c, t_start, config, path=PATH, filename=name)
        if EVAL:
            mass = evalTotalMass(c, dx, dy, dz)
            print(f"Total mass at time {t_start}: {mass:.6f}")
        print(f"Frame {t_start} written in time_{t_start:06}.vtk.")

    # ----------------- TIME LOOP ----------------- #
    if EVAL:
        mass = torch.zeros(nb_steps, dtype=torch.float32, device='cpu')
        energy = torch.zeros(nb_steps, dtype=torch.float32, device='cpu')

    with torch.no_grad():
        for t in tqdm(range(t_start+1, t_start+nb_steps+1), desc="Solving..."):
            # compute g
            compute_g(g_device, c_device, A)

            # compute phi
            compute_vm(phi_device, c_device, M)

            torch.complex(g_device, zeros_device, out=g_device_fft)
            torch.complex(c_device, zeros_device, out=c_device_fft)

            torch.fft.fftn(g_device_fft, out=g_device_fft)
            torch.fft.fftn(c_device_fft, out=c_device_fft)

            # compute r
            compute_r(r_device_fft, c_device_fft, g_device_fft, kappa, k2)

            # compute y
            compute_y((y1_device_fft, y2_device_fft, y3_device_fft), r_device_fft, kx_mesh, ky_mesh, kz_mesh)

            # bring back to real space
            y1_device.copy_(torch.fft.ifftn(y1_device_fft, dim=(0, 1, 2), out=y1_device_fft).real)
            y2_device.copy_(torch.fft.ifftn(y2_device_fft, dim=(0, 1, 2), out=y2_device_fft).real)
            y3_device.copy_(torch.fft.ifftn(y3_device_fft, dim=(0, 1, 2), out=y3_device_fft).real)

            # compute q
            compute_q((q1_device_fft, q2_device_fft, q3_device_fft), y1_device, y2_device, y3_device, phi_device)

            torch.fft.fftn(q1_device_fft, out=q1_device_fft)
            torch.fft.fftn(q2_device_fft, out=q2_device_fft)
            torch.fft.fftn(q3_device_fft, out=q3_device_fft)

            # update c
            solve_ch(c_device_fft, q1_device_fft, q2_device_fft, q3_device_fft, kx_mesh, ky_mesh, kz_mesh, k4, kappa, alpha, dt)

            # bring back to real space
            c_device.copy_(torch.fft.ifftn(c_device_fft, dim=(0, 1, 2), out=c_device_fft).real)

            if EVAL:
                mass[t-1] = evalTotalMass(c_device, dx, dy, dz)
                energy[t-1] = evalTotalEnergy(c_device, c_device_fft, A, kappa, kx_mesh, ky_mesh, kz_mesh, dx, dy, dz)

            if SAVE:
                saveFrame(c_device, t, freq, config, path=PATH, filename=name)

    # ----------------- POST-PROCESSING ----------------- #
    if EVAL:
        mass_filename = f"mass_{t_start}-{t_end}_{dt}"
        energy_filename = f"energy_{t_start}-{t_end}_{dt}"
        t_arr = np.arange(t_start, t_end, dt)
        torch.save(mass, f"{PATH}{mass_filename}.pt")
        torch.save(energy, f"{PATH}{energy_filename}.pt")

    # pvd_filename = f"{PATH}simulation.pvd"
    # create_binary_pvd(PATH, pvd_filename=pvd_filename)
    # os.system(f"paraview --data={PATH}{pvd_filename}")

    

    # clear memory
    del kx_mesh, ky_mesh, kz_mesh, k2, k4
    del c_device, zeros_device, g_device, phi_device, y1_device, y2_device, y3_device
    del c_device_fft, g_device_fft, r_device_fft, y1_device_fft, y2_device_fft, y3_device_fft
    del q1_device_fft, q2_device_fft, q3_device_fft

    Device.empty_cache()
