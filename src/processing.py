import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gmsh
import sys
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


class MyDevice:
    """
    A class used to represent a Device.

    :param cpu: bool, optional, if True, the device will be set to CPU, defaults to False
    """

    device_name = ""
    sync_method = None

    def __init__(self, cpu=False):
        if cpu:
            self.device_name = "cpu"
            self.sync_method = lambda: None
            self.empty_cache = lambda: None
        else:
            if torch.cuda.is_available():
                self.device_name = "cuda"
                self.sync_method = lambda: torch.cuda.synchronize()
                self.empty_cache = lambda: torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                self.device_name = "mps"
                self.sync_method = lambda: torch.mps.synchronize()
                self.empty_cache = lambda: torch.mps.empty_cache()
            else:
                # raise warning in console
                print("Warning: neither cuda nor mps are available, using cpu instead...")
                self.device_name = "cpu"
                self.sync_method = lambda: None

    def synchronize(self):
        """
        Synchronize the device.

        :return: None
        """
        self.sync_method()

    def empty_cache(self):
        """
        Empty the cache of the device.

        :return: None
        """
        self.empty_cache()


class EdgeDetection(nn.Module):
    def __init__(self):
        super(EdgeDetection, self).__init__()
        # Define a 3x3x3 kernel for the erosion operation
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
        self.register_buffer('kernel', kernel)

    def forward(self, volume):
        # Add batch and channel dimensions to the volume
        volume = volume.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, nx, ny, nz)

        # Apply 3D erosion using 3D convolution with the registered kernel
        eroded_volume = F.conv3d(volume, self.kernel, padding=1)  # Ensure same size output
        eroded_volume = ( eroded_volume == self.kernel.sum()).float()  # Binary eroded volume

        # Subtract the eroded volume from the original volume to get the boundary
        boundary = volume - eroded_volume

        # Set the boundary values to 0.5
        volume = volume.squeeze()  # Remove batch and channel dimensions
        boundary = boundary.squeeze()  # Remove batch and channel dimensions
        volume[boundary == 1.0] = 0.5

        return volume


def loadGeom(filename: str):
    """
    Load the geometry from a file.

    :param filename: str, the name of the file

    :return: dict, the configuration parameters
    """
    config = {}
    # deprecated
    # if ANGLE_FLAG:
    #     gmsh.initialize(argv=["", "-bin"])
    #     gmsh.option.setNumber("General.Terminal", 0)

    #     gmsh.open(filename)

    #     gmsh.onelab.setNumber("Parameters/angle", np.array([angle], dtype=np.float64))

    #     gmsh.model.geo.synchronize()
    #     gmsh.model.occ.synchronize()
    #     gmsh.finalize()

    # initialize gmsh and generate the mesh
    gmsh.initialize(argv=["", "-bin"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(filename)
    gmsh.model.mesh.generate(3)

    # read .geo file
    config["Nx"] = int(gmsh.onelab.getNumber("Parameters/Nx")[0])
    config["Ny"] = int(gmsh.onelab.getNumber("Parameters/Ny")[0])
    config["Nz"] = int(gmsh.onelab.getNumber("Parameters/Nz")[0])

    config["Lx"] = gmsh.onelab.getNumber("Parameters/Lx")[0]
    config["Ly"] = gmsh.onelab.getNumber("Parameters/Ly")[0]
    config["Lz"] = gmsh.onelab.getNumber("Parameters/Lz")[0]

    config["nb_nw"] = int(gmsh.onelab.getNumber("Parameters/Nanowire number")[0]) + 1
    config["angle"] = gmsh.onelab.getNumber("Parameters/angle")[0]

    config["dX"] = gmsh.onelab.getNumber("Parameters/dX")[0]

    config["r_1"] = gmsh.onelab.getNumber("Parameters/r_1")[0]
    config["r_2"] = gmsh.onelab.getNumber("Parameters/r_2")[0]

    return config


def sortByCoords(tensor: torch.Tensor, coords: torch.Tensor):
    """
    Sort a tensor and its coordinates based on the coordinates.

    :param tensor: torch.Tensor, the tensor to sort
    :param coords: torch.Tensor, the coordinates to sort

    :return: tuple, the sorted tensor and the sorted coordinates
    """

    Nx, Ny, Nz = tensor.shape

    coords_flatten = coords.view(-1, 3)
    tensor_flatten = tensor.view(-1,)

    x = coords_flatten[:, 0]
    x, indices = torch.sort(x)
    coords_flatten = coords_flatten[indices]
    tensor_flatten = tensor_flatten[indices]

    for i in tqdm(range(0, Nx), desc="Sorting..."):
        # take a slice
        x_slice = coords_flatten[i*Ny*Nz:(i+1)*Ny*Nz]
        idx_slice = indices[i*Ny*Nz:(i+1)*Ny*Nz]
        tensor_slice = tensor_flatten[i*Ny*Nz:(i+1)*Ny*Nz]
        # sort the slice
        y = x_slice[:, 1]
        y, indices_y = torch.sort(y)
        # apply the same permutation to the other arrays
        x_slice = x_slice[indices_y]
        idx_slice = idx_slice[indices_y]
        tensor_slice = tensor_slice[indices_y]
        for j in range(0, Ny):
            # take a slice
            y_slice = x_slice[j*Nz:(j+1)*Nz]
            idx_slice_2 = idx_slice[j*Nz:(j+1)*Nz]
            tensor_slice_2 = tensor_slice[j*Nz:(j+1)*Nz]
            # sort the slice
            z = y_slice[:, 2]
            z, indices_z = torch.sort(z)
            # apply the same permutation to the other arrays
            y_slice = y_slice[indices_z]
            idx_slice_2 = idx_slice_2[indices_z]
            tensor_slice_2 = tensor_slice_2[indices_z]
            # update the slice
            x_slice[j*Nz:(j+1)*Nz] = y_slice
            idx_slice[j*Nz:(j+1)*Nz] = idx_slice_2
            tensor_slice[j*Nz:(j+1)*Nz] = tensor_slice_2
        # update the slice
        coords_flatten[i*Ny*Nz:(i+1)*Ny*Nz] = x_slice
        indices[i*Ny*Nz:(i+1)*Ny*Nz] = idx_slice
        tensor_flatten[i*Ny*Nz:(i+1)*Ny*Nz] = tensor_slice

    sorted_coords = coords_flatten.view((Nx,Ny,Nz, 3))
    sorted_indices = indices.view((Nx,Ny,Nz,))
    sorted_tensor = tensor_flatten.view((Nx,Ny,Nz,))

    return sorted_tensor, sorted_coords


def makeCompositionField(meshgrid_size: "tuple[int, int, int]", geom_dim: "tuple[int, int, int]", c_zero: float):
    """
    Make the composition field.

    :param meshgrid_size: tuple[int], the size of the meshgrid
    :param geom_dim: tuple[int], the dimensions of the geometry
    :param c_zero: float, the value of the composition field in the physical group

    :return: tuple, the composition field and the coordinates
    """

    # fetch physical groups
    dimTags = gmsh.model.getPhysicalGroups()

    dimTag = dimTags[-1]

    # set meshgrid size and geometry dimensions
    nx, ny, nz = meshgrid_size
    Lx, Ly, Lz = geom_dim

    dx = Lx / float(nx)
    dy = Ly / float(ny)
    dz = Lz / float(nz)

    c = torch.zeros((nx, ny, nz), dtype=torch.float32)

    x = torch.arange(start=0.0, end=Lx, step=dx)
    y = torch.arange(start=0.0, end=Ly, step=dy)
    z = torch.arange(start=0.0, end=Lz, step=dz)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

    coords = torch.stack((grid_x, grid_y, grid_z), dim=3)

    # flatten the composition field and the coordinates
    coords = coords.view(-1, 3)
    c = c.view(-1,)

    # get the entities of the physical group
    vols = gmsh.model.getEntitiesForPhysicalGroup(dimTags[0][0], dimTags[0][1])
    phys_ele = []
    for vol in vols:
        out = gmsh.model.mesh.getElements(dimTags[0][0], vol)
        phys_ele.extend(out[1][0])

    phys_ele = torch.tensor(phys_ele, dtype=torch.int32).view(-1,)

    n_nodes = nx*ny*nz

    check = {}
    for n in tqdm(range(n_nodes), desc="Setting composition field..."):
        x, y, z = coords[n]
        el_tag, _, _, _, _, _ = gmsh.model.mesh.getElementByCoordinates(x, y, z, dim=3)
        
        if check.get(el_tag) is True:
            c[n] = c_zero
        elif check.get(el_tag) is None:
            if el_tag in phys_ele:
                c[n] = c_zero
                check[el_tag] = True
            else:
                check[el_tag] = False
        else:
            continue

    # reshape the composition field and the coordinates
    c = c.view((nx, ny, nz))
    coords = coords.view((nx, ny, nz, 3))
    gmsh.finalize()
    return c, coords


def _process_chunk(args, file):
    start_idx, end_idx, coords, phys_ele, c_zero, idx = args
    print(f"Thread {idx} processing chunk {idx}...")
    loadGeom(file)

    local_c = torch.zeros(end_idx - start_idx, dtype=torch.float32)
    chunck_size = end_idx - start_idx
    #with tqdm(total=chunck_size, desc=f"Thread {idx}", position=idx, leave=True) as pbar:
    for i in range(start_idx, end_idx):
        x, y, z = coords[i]
        el_tag, _, _, _, _, _ = gmsh.model.mesh.getElementByCoordinates(x, y, z, dim=3)
        if el_tag in phys_ele:
            local_c[i - start_idx] = c_zero

    return local_c.numpy()


def makeCompositionFieldParallel(meshgrid_size: "tuple[int, int, int]", geom_dim: "tuple[int, int, int]", c_zero: float, file: str, num_workers=8):
    """
    Make the composition field in parallel.

    :param meshgrid_size: tuple[int], the size of the meshgrid
    :param geom_dim: tuple[int], the dimensions of the geometry
    :param c_zero: float, the value of the composition field in the physical group
    :param file: str, the name of the file

    :return: tuple, the composition field and the coordinates
    """

    dimTags = gmsh.model.getPhysicalGroups()

    nx, ny, nz = meshgrid_size
    Lx, Ly, Lz = geom_dim
    dx = Lx / float(nx)
    dy = Ly / float(ny)
    dz = Lz / float(nz)

    x = torch.arange(start=0.0, end=Lx, step=dx)
    y = torch.arange(start=0.0, end=Ly, step=dy)
    z = torch.arange(start=0.0, end=Lz, step=dz)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack((grid_x, grid_y, grid_z), dim=3).view(-1, 3)

    vols = gmsh.model.getEntitiesForPhysicalGroup(dimTags[0][0], dimTags[0][1])
    phys_ele = []
    for vol in vols:
        out = gmsh.model.mesh.getElements(dimTags[0][0], vol)
        phys_ele.extend(out[1][0])
    phys_ele = np.array(phys_ele, dtype=np.int32)

    n_nodes = nx * ny * nz
    chunk_size = n_nodes // num_workers
    chunk_args = [(i, min(i + chunk_size, n_nodes), coords.numpy(), phys_ele, c_zero, idx+1) for idx, i in enumerate(range(0, n_nodes, chunk_size))]

    results = []
    with Pool(num_workers) as pool:
        with tqdm(total=len(chunk_args), desc="Processing mesh chunks") as pbar:
            results = pool.starmap(_process_chunk, [(args, file) for args in chunk_args])
            pbar.update(len(chunk_args))
        pool.close()
        pool.join()

    # Combine results
    c = torch.cat([torch.tensor(result) for result in results]).view(nx, ny, nz)

    gmsh.finalize()

    return c, coords.view((nx, ny, nz, 3))


def makeIndicatorFunction(Lx, Ly, Lz, dx, dy, dz, substrate_dim, type="circular"):
    # define indicator function
    x1 = Lx//2
    y1 = Ly//2
    z1 = Lz//2
    _x = torch.arange(start=0.0, end=Lx, step=dx)
    _y = torch.arange(start=0.0, end=Ly, step=dy)
    _z = torch.arange(start=0.0, end=Lz, step=dz)
    h = 2
    tanh_inv = np.arctanh(0.9)
    eps = dx*h/(4*np.sqrt(2)*tanh_inv)

    xx, yy, zz = torch.meshgrid(_x, _y, _z, indexing='ij')
    if type == "planar":
        indicator = 1/2 * (1+torch.tanh((Ly-Ly//2-substrate_dim-yy)/(eps)))
    elif type == "circular":
        indicator = 1/2 * (1+torch.tanh(-(substrate_dim-torch.sqrt((xx-x1)**2+(yy-y1)**2+(zz-z1)**2))/(eps)))
    else :
        raise ValueError("Invalid type of indicator function. Choose between 'planar' and 'circular'.")

    return indicator


def writeVTK(data: torch.Tensor, t: int, config: dict, path="output/", filename="time"):
    """
    Write the data to a VTK file.

    :param data: torch.Tensor, the data to write
    :param t: int, the time step
    :param config: dict, the configuration parameters
    :param path: str, the path to save the file
    :param filename: str, the name of the file

    :return: None
    """

    data = data.numpy()
    nx, ny, nz = data.shape
    # spacing = (1, 1, 1)
    lx, ly, lz = config["Lx"], config["Ly"], config["Lz"]
    spacing = (config["dX"], config["dX"], config["dX"])
    origin = (0, 0, 0)

    filename = f"{path}{filename}_{t:06}.vtk"
    config_str = f",".join([f"{key}:{value}" for key, value in config.items()])
    config_line = f"DESC composition field at time {t} ; CONFIG {config_str}"
    # write in one go the header
    header = (f"# vtk DataFile Version 3.0\n"
                f"{config_line}\n"
                f"BINARY\n"
                f"DATASET STRUCTURED_POINTS\n"
                f"DIMENSIONS {nx} {ny} {nz}\n"
                f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
                f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n"
                f"POINT_DATA {nx*ny*nz}\n"
                f"SCALARS density_field float\n"
                f"LOOKUP_TABLE default\n")

    with open(filename, 'wb') as file:
        file.write(header.encode())
        flat_data = np.ravel(data, order='F').astype('>f4')
        file.write(flat_data.tobytes())
    return


def readVTK(filename: str):
    """
    Read a VTK file.

    :param filename: str, the name of the file

    :return: tuple[np.ndarray, dict], the data and the configuration parameters
    """
    config = {}
    with open(filename, 'rb') as file:
        # Read and decode the header information
        header = bytearray()
        file.readline()  # Skip the first line 
        while True:
            line = file.readline()
            header += line
            if b"LOOKUP_TABLE default\n" in line:
                break  # Stop after the header ends

        # Decode header to extract dimensions
        header_text = header.decode('ascii')
        lines = header_text.split('\n')
        for line in lines:
            if "CONFIG" in line:
                configuration = line.split(';')[1:]
                keys_values_str = configuration[0].split()[1:]
                key_values_arr = keys_values_str[0].split(',')
                for kv in key_values_arr:
                    key, value = kv.split(':')
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                    config[key] = value
            if "DIMENSIONS" in line:
                nx = int(line.split()[1])
                ny = int(line.split()[2])
                nz = int(line.split()[3])


        # Read the binary data
        data_flat = np.fromfile(file, dtype='>f4').astype(np.float32)  # Big-endian double precision float -> np.float32
        data = data_flat.reshape((nx, ny, nz), order='F')
        data = np.ascontiguousarray(data)

    return data, config


def saveFrame(c_device: torch.Tensor, t: int, save_every, config: dict, dt: float, path="output/", filename="time"):
    """
    Save a frame to a VTK file.

    :param c_device: torch.Tensor, the composition field
    :param t: Any, the time step
    :param save_every: int or list[int], save frequency
    :param config: dict, the configuration parameters
    :param path: str, the path to save the file

    :return: None
    """
    if isinstance(save_every, list):
        if t == 0:
            pass
        elif t == save_every[0]:
            save_every.pop(0)
            pass
        else:
            return
    elif t % save_every == 0:
        pass
    else:
        return
    
    c = c_device.cpu()
    mass = evalTotalMass(c)
    print(f"Total mass at time {t}: {mass:.6f}")
    writeVTK(c, t, config, path=path, filename=filename)
    print(f"Frame {t} written in {path}{filename}_{t:06}.vtk.")


def plotComp(c: torch.Tensor):
    nx, ny, nz = c.shape
    freqs = np.linspace(0, nz-1, num=5, dtype=int)
    for i in freqs:
        plt.imshow(c[:, :, i], cmap='gray')
        plt.title(f"Slice {i}")
        plt.imsave(f"output/slice/slice_{i}.png", c[:, :, i], cmap='gray')

    return


def evalTotalMass(c_device: torch.Tensor):
    """
    Evaluate the mass of the composition field.

    :param c: torch.Tensor, the composition field
    :param dx: float, the step in x
    :param dy: float, the step in y
    :param dz: float, the step in z

    :return: float, the mass of the composition field
    """
    shape = c_device.shape
    N = shape[0]*shape[1]*shape[2]
    return (torch.sum(c_device)/N).item()


def evalTotalEnergy(c_device, c_device_fft, A, kappa, kx, ky, kz, dx):
    """
    Evaluate the total energy of the composition field.

    :param c: torch.Tensor, the composition field

    :return: float, the total energy of the composition field
    """
    # evaluate the bulk free energy (double well potential)
    f_0 = A*c_device*c_device*(1.0 - c_device)*(1.0 - c_device)

    torch.fft.rfftn(c_device, out=c_device_fft, dim=(0, 1, 2))

    # compute the gradient of the composition field
    grad_c_1 = torch.fft.ifftn(1j*kx*c_device_fft, dim=(0, 1, 2)).real
    grad_c_2 = torch.fft.ifftn(1j*ky*c_device_fft, dim=(0, 1, 2)).real
    grad_c_3 = torch.fft.ifftn(1j*kz*c_device_fft, dim=(0, 1, 2)).real

    # compute square of the gradient magnitude
    grad_c = torch.sqrt(grad_c_1**2 + grad_c_2**2 + grad_c_3**2)
    grad_c_squared = grad_c**2 

    # compute the interfacial energy
    f_int = kappa/2*grad_c_squared


    N = c_device.shape[0]*c_device.shape[1]*c_device.shape[2]
    # compute the total energy
    total_energy = (torch.sum(f_0 + f_int)).item()/N
    
    return total_energy



# deprecated
# def makeCompositionField(meshgrid_size, c_zero):
#     ## FIRST METHOD (extremely slow but accurate)
#     nx, ny, nz = meshgrid_size
#     c = torch.zeros((nx * ny * nz, 1), dtype=torch.float32)

#     # get physical groups
#     dimTags = gmsh.model.getPhysicalGroups()
#     dimTag = dimTags[-1]

#     vols = gmsh.model.getEntitiesForPhysicalGroup(dimTag[0], dimTag[1])

#     # get all the nodes
#     nodes = gmsh.model.mesh.getNodes(returnParametricCoord=False)

#     # create a composition field
#     c = torch.zeros(len(nodes[0]), dtype=torch.float32)

#     for i in tqdm(range(len(nodes[0])), desc="Checking if nodes are inside the domain..."):
#         x, y, z = nodes[1][i * 3], nodes[1][i * 3 + 1], nodes[1][i * 3 + 2]
#         for vol in vols:
#             if gmsh.model.isInside(dimTag[0], vol, [x, y, z]):
#                 c[i] = c_zero

#     # get x, y and z coords
#     coords = np.array([nodes[1][i * 3:i * 3 + 3] for i in range(0, len(nodes[0]))])
#     coords = torch.tensor(coords, dtype=torch.float32)
#     coords = coords.view((nx, ny, nz, 3))
#     c = c.view((nx, ny, nz))

#     return c, coords, nodes
    

# def writeVTK(data, t, path="output/", filename="time"):
#     data = data.numpy()
#     nx, ny, nz = data.shape
#     spacing = (1, 1, 1)
#     origin = (0, 0, 0)

#     filename = f"{path}{filename}_{t:06}.vtk"

#     with open(filename, 'w') as file:
#         file.write(f"# vtk DataFile Version 3.0\n")
#         file.write(f"composition field at time {t}\n")
#         file.write(f"ASCII\n")
#         file.write(f"DATASET STRUCTURED_POINTS\n")
#         file.write(f"DIMENSIONS {nx} {ny} {nz}\n")
#         file.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
#         file.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n")
#         file.write(f"POINT_DATA {nx*ny*nz}\n")
#         file.write(f"SCALARS density_field double\n")
#         file.write(f"LOOKUP_TABLE default\n")
#         flat_data = np.ravel(data, order='F')
#         formatted_data = f'\n'.join(map(str, flat_data))
#         file.write(formatted_data)
#         # for k in range(nz):
#         #     for j in range(ny):
#         #         for i in range(nx):
#         #             file.write(f"{data[i, j, k]}\n")
#     return


# deprecated
# def readVTK(filename, format="ascii"):
#     out_dict = {}
#     mode = "r" if format == "ascii" else "rb"
#     with open(filename, mode) as file:
#         lines = file.readlines()
#         nx, ny, nz = [int(i) for i in lines[4].split()[1:]]
#         data = np.zeros((nx, ny, nz))
#         for k in tqdm(range(nz), desc="Reading VTK file..."):
#             for j in range(ny):
#                 for i in range(nx):
#                     data_point = lines[10 + i + j*nx + k*nx*ny]
#                     data[i, j, k] = float(data_point)

#     #out_dict["c"] = data
#     return data


# deprecated
# def loadGeom(filename, angle):
#     gmsh.initialize()
#     gmsh.option.setNumber("General.Terminal", 0)

#     gmsh.open(filename)

#     gmsh.onelab.setNumber("Parameters/angle", np.array([angle], dtype=np.float64))
#     gmsh.onelab.setNumber("Parameters/MESH", np.array([1], dtype=np.float64))
#     gmsh.model.geo.synchronize()
#     gmsh.model.occ.synchronize()
#     gmsh.finalize()
#     # print(".geo set up done in ", time.time() - tread, "seconds.")
#     # read .geo file
#     # tread = time.time()
#     gmsh.initialize()
#     gmsh.option.setNumber("General.Terminal", 0)

#     gmsh.open(filename)