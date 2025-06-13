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
from functools import lru_cache

from utils import evalTotalMass

FRAME_NB = 0

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

    config["type"] = f'nanowire'
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


def makeCompositionField(meshgrid_size: "tuple[int, int, int]", 
                         geom_dim: "tuple[int, int, int]", c_zero: float):
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
    #c_alpha, c_beta = c_eq
    # each thread initialize instance of gmsh, reads the geometry file and mesh it
    loadGeom(file)

    test_tag = lru_cache(lambda el_tag : _tag_in_phys_group(el_tag, phys_ele))

    local_c = torch.zeros(end_idx - start_idx, dtype=torch.float32)#.mul_(c_beta)
    chunck_size = end_idx - start_idx
    for i in range(start_idx, end_idx):
        x, y, z = coords[i]
        el_tag, _, _, _, _, _ = gmsh.model.mesh.getElementByCoordinates(x, y, z, dim=3, strict=False)
        if test_tag(el_tag):
            local_c[i - start_idx] = c_zero

    return local_c.numpy()

def _tag_in_phys_group(el_tag, phys_ele):
    if el_tag in phys_ele:
        return True
    else:
        return False


def makeCompositionFieldParallel(meshgrid_size: "list[int]", 
                                 geom_dim: "list[float]", c_zero: float, file: str, num_workers=8):
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


def makeCompositionFieldTorch(config, kappa, A, shape=0, k_mode=0, L1=None, L2=None):
    Lx, Ly, Lz = config["Lx"], config["Ly"], config["Lz"]
    nx, ny, nz = config["Nx"], config["Ny"], config["Nz"]
    dx = config["dX"]

    R1 = config["r_1"]*dx
    if L1 is None:
        L1 = 2.0 * Lz

    # define the meshgrid
    x, y, z = torch.meshgrid(torch.linspace(0.0, Lx, nx), torch.linspace(0.0, Ly, ny), torch.linspace(0.0, Lz, nz), indexing='ij')

    # define the composition field
    eps = np.sqrt(2.0*kappa/A)

    d = np.float32(4.185*eps)

    midplane = torch.tensor([Lx/2.0, Ly/2.0, 0.0])
    c1 = torch.ones((nx, ny, nz), dtype=torch.float32)
    if shape == 1:
        R1_pent = np.pi/(5.0 * np.sin(np.pi/5.0)) * R1 # outer-surface matching
        R1 = R1_pent

        prfl1 = torch.ones((nx, ny, nz), dtype=torch.float32)
        alpha = 2*np.pi/5
        offset = alpha + np.pi/2
        for i in range(5): # pentagonal cross-section
            p1 = midplane + R1*torch.tensor([np.cos(i*alpha + offset), np.sin(i*alpha + offset), 0.0], dtype=torch.float32)
            if i < 4:
                p2 = midplane + R1*torch.tensor([np.cos((i+1)*alpha + offset), np.sin((i+1)*alpha + offset), 0.0], dtype=torch.float32)
            else:
                p2 = midplane + R1*torch.tensor([np.cos(0 + offset), np.sin(0 + offset), 0.0], dtype=torch.float32)
            
            a = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - a*p1[0]

            prfl1 = prfl1 * (a*x + b - y)

        c1 = 1/2 * (1 + torch.tanh(-prfl1/eps)) # pentagonal cross-section
    
    noise = torch.zeros((nx, ny, nz), dtype=torch.float32)
    # ampl = 0.5*10.0
    ampl = 5e-2
    K = 10
    k_arr = np.linspace(2.0*np.pi/Lz, 2.0*np.pi/(7*R1), K)
    # for i, k in enumerate(k_arr):
        # noise += ampl/K * torch.sin(2.0*np.pi*z / (k*R1))
        # noise += ampl/K * torch.sin(k * z)
    noise = ampl * torch.sin(k_arr[k_mode]*z)
    
    torch.random.manual_seed(config["seed"])
    noise = ampl * (0.5-torch.rand(*x.shape, dtype=torch.float32))
    # noise = ampl * torch.sin(2.0*np.pi*z / (11.0*R1))
    c1.mul_(1/2 * (1 + torch.tanh(((R1*(1+noise))-torch.sqrt((x-midplane[0])**2 + (y-midplane[1])**2))/eps))) # limit to interior of circumscribed cylinder

    if config["nb_nw"] > 1:
        theta = -np.deg2rad(config["angle"])
        rotation_axis = torch.tensor([Lx/2.0, Ly/2.0, Lz/2.0])
        R2 = config["r_2"]*dx

        # define rotated coordinates
        xx = np.cos(theta)*(x - rotation_axis[0]) + np.sin(theta)*(z - rotation_axis[2]) + rotation_axis[0]
        yy = y
        zz = -np.sin(theta)*(x - rotation_axis[0]) + np.cos(theta)*(z - rotation_axis[2]) + rotation_axis[2]

        c2 = torch.ones((nx, ny, nz), dtype=torch.float32)
        y_offset = torch.tensor([0.0, R1+R2+d, 0.0])
        if shape == 1:
            R2_pent = np.pi/(5.0 * np.sin(np.pi/5.0)) * R2 # outer-surface matching
            R2 = R2_pent

            prfl2 = torch.ones((nx, ny, nz), dtype=torch.float32)
            alpha = 2*np.pi/5
            offset = alpha + np.pi/2
            for i in range(5): # pentagonal cross-section
                p1 = midplane + y_offset + R2*torch.tensor([np.cos(i*alpha + offset), np.sin(i*alpha + offset), 0.0], dtype=torch.float32)
                if i < 4:
                    p2 = midplane + y_offset + R2*torch.tensor([np.cos((i+1)*alpha + offset), np.sin((i+1)*alpha + offset), 0.0], dtype=torch.float32)
                else:
                    p2 = midplane + y_offset + R2*torch.tensor([np.cos(0 + offset), np.sin(0 + offset), 0.0], dtype=torch.float32)
                
                a = (p2[1] - p1[1]) / (p2[0] - p1[0])
                b = p1[1] - a*p1[0]

                prfl2 = prfl2 * (a*xx + b - yy)

            c2 = 1/2 * (1 + torch.tanh(-prfl2/eps)) # pentagonal cross-section

        dZ = 3*R1
        Z1, Z2 = Lz/2-3*R1, Lz/2+3*R1
        dY = (R1 + R2 + d + 4*dx)/2 * (torch.tanh((zz - Z1)/dZ) - torch.tanh((zz - Z2)/dZ))
        # c2.mul_(1/2 * (1 + torch.tanh((R2-torch.sqrt((xx-midplane[0])**2 + (yy-midplane[1] - y_offset[1])**2))/eps))) # limit to interior of circumscribed cylinder
        c2.mul_(1/2 * (1 + torch.tanh(((R2+noise)-torch.sqrt((xx-midplane[0])**2 + (yy-midplane[1] - dY)**2))/eps))) # limit to interior of circumscribed cylinder

        return c1 + c2
    else:
        return c1
    
def sdf_circle(x, y, z, cx, cy, cz, r, L, theta=0):
    """
    Signed distance function for a circle/cylinder centered at (cx, cy) with radius r/and z-wire axis rotated along y axis at cz.
    
    Parameters:
        x, y, z : Coordinates (scalars or arrays)
        cx, cy : Center of the circle
        cz : position of rotation axis
        r : Radius of the circle
        L : Length of the cylinder along z-axis
        theta: Rotation angle along y
        
    Returns:
        Signed distance: < 0 inside, = 0 on edge, > 0 outside
    """
    if theta != 0:
        # rotate along y
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_rot = cos_theta * (x - cx) + sin_theta * (z - cz)
        z_rot = -sin_theta * (x - cx) + cos_theta * (z - cz)
        x, z = x_rot + cx, z_rot + cz

    return np.sqrt((x - cx)**2 + (y - cy)**2) - r
    
def sdf_ngon(x, y, cx, cy, r, num_sides=5):
    """
    Computes the signed distance function from a regular n-gon (default: pentagon),
    centered at (cx, cy), with circumradius r.
    
    Parameters:
        x, y, z         : Grid coordinates (can be scalars or arrays)
        cx, cy          : Center of the polygon
        r               : Radius from center to vertices (circumradius)
        num_sides       : Number of polygon sides (e.g., 5 for pentagon).
        
    Returns:
        d               : Signed distance to the polygon (negative inside, zero on edge, positive outside)
    """
    # Translate to center
    px = x - cx
    py = y - cy

    # Rotate coordinates
    cos_a = np.cos(-np.pi / 2 - np.pi/5.0)
    sin_a = np.sin(-np.pi / 2 - np.pi/5.0)
    xr = cos_a * px - sin_a * py
    yr = sin_a * px + cos_a * py

    # Convert to polar
    theta = np.arctan2(yr, xr)
    rho = np.hypot(xr, yr)

    # Polygon angle logic
    angle = 2 * np.pi / num_sides
    half_angle = angle / 2

    # Map angle into wedge
    theta_mod = np.mod(theta + half_angle, angle) - half_angle

    # Compute distance to edge
    d = rho * np.cos(theta_mod) - r * np.cos(np.pi / num_sides)
    return d


def makeIndicatorFunction(meshgrid_size:"list[int]", 
                          geom_dim:"list[float]", A, kappa, substrate_dim, type="circular"):
    # fetch dimensions
    nx, ny, nz = meshgrid_size
    Lx, Ly, Lz = geom_dim
    dx = Lx / float(nx)
    dy = Ly / float(ny)
    dz = Lz / float(nz)

    # fetch parameters
    

    # define indicator function
    x1 = Lx//2
    y1 = Ly//2
    z1 = Lz//2

    # wall thickness
    # thin wall 0.1
    # medium wall 0.2
    # thick wall 0.3
    # ratio = 0.3 # ratio of the wall thickness to the charac length
    # t = (L_star * ratio) # wall thickness
    # eps = (t)/4.185 # 
    # eps = np.sqrt(2*dx)/4.185
    # eps = np.sqrt(2*dx)*4
    # eps = np.sqrt(2*dx)
    # eps = np.sqrt(2*dx)/4.1285
    N = 1.0
    # eps = N * dx / 4.185
    eps = np.sqrt(2.0*kappa/A)

    _x = torch.linspace(start=0.0, end=Lx, steps=nx)
    _y = torch.linspace(start=0.0, end=Ly, steps=ny)
    _z = torch.linspace(start=0.0, end=Lz, steps=nz)

    # h = 40
    # tanh_inv = np.arctanh(0.9)
    # eps = dx*h/(4*np.sqrt(2)*tanh_inv)

    if nz > 1:
        xx, yy, zz = torch.meshgrid(_x, _y, _z, indexing='ij')
    else:
        xx, yy = torch.meshgrid(_x, _y, indexing='ij')
        zz = torch.zeros_like(xx)

    if type == "planar":
        # indicator = 1/2 * (1 + torch.tanh((Ly-Ly//2-substrate_dim-yy)/(eps)))
        indicator = 1/2 * (1 + (torch.tanh((yy-substrate_dim)/eps)))# - 1/2 * (1 + torch.tanh((yy-3/4*Ly)/eps))
        # indicator = 1/2 * (1 + torch.tanh(((xx - Lx/10)/eps)))
        # indicator = 1 - 1/2 * (torch.tanh((yy-substrate_dim+3*4.185*eps)/eps) - (torch.tanh((yy-substrate_dim)/eps)))
        # indicator[indicator >= 0.5] = 1.0
        # indicator[indicator < 0.5] = 0.0
    elif type == "circular":
        indicator = 1/2 * (1 + torch.tanh((substrate_dim-torch.sqrt((xx-x1)**2+(yy-y1)**2+(zz-z1)**2))/(eps)))
    elif type == "square":
        a = 20.0
        R = Lx/2
        indicator = 1/2 * (1 + torch.tanh((R - torch.pow((xx-Lx/2)**a + (yy-Ly/2)**a + (zz-Lz/2)**a, 1/a))/eps))
    else :
        raise ValueError("Invalid type of indicator function. Choose between 'planar' and 'circular'.")

    return indicator


def writeVTK(data: torch.Tensor, t: any, config: dict, path="output/", filename="time",  desc="density_field"):
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
    if data.ndim > 2:
        nx, ny, nz = data.shape
    else:
        nx, ny = data.shape
        nz = 1
    # spacing = (1, 1, 1)
    lx, ly, lz = config["Lx"], config["Ly"], config["Lz"]
    spacing = (config["dX"], config["dX"], config["dX"])
    origin = (0, 0, 0)

    # filename = f"{path}{filename}_{t:08.1f}_{FRAME_NB}.vtk"
    filename = f"{path}{filename}_{FRAME_NB}.vtk"
    config_str = f",".join([f"{key}:{value}" for key, value in config.items()])
    config_line = f"DESC {desc} at time {t:08.2f} ; CONFIG {config_str}"
    # write in one go the header
    header = (f"# vtk DataFile Version 3.0\n"
                f"{config_line}\n"
                f"BINARY\n"
                f"DATASET STRUCTURED_POINTS\n"
                f"DIMENSIONS {nx} {ny} {nz}\n"
                f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
                f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n"
                f"POINT_DATA {nx*ny*nz}\n"
                f"SCALARS {desc} float\n"
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
        if nz > 1:
            data = data_flat.reshape((nx, ny, nz), order='F')
        else:
            data = data_flat.reshape((nx, ny), order='F')
        data = np.ascontiguousarray(data)

    return data, config


def saveFrame(c_device: torch.Tensor, 
              t: int, save_every, config: dict, dt: float, path="output/", filename="time", desc="density_field", t_elapsed=0.0, eval=True):
    """
    Save a frame to a VTK file.

    :param c_device: torch.Tensor, the composition field
    :param t: Any, the time step
    :param save_every: int or list[int], save frequency
    :param config: dict, the configuration parameters
    :param path: str, the path to save the file

    :return: None
    """
    if isinstance(save_every, np.ndarray):
        save_every_array = np.abs(save_every - t_elapsed)
        if np.any(save_every_array < dt):
            pass
        elif t == 0:
            pass
        #elif t == save_every[0]:
        #    save_every.pop(0)
        #    pass
        # elif t in save_every:
        #     pass
        else:
            return
    elif t % int(save_every) == 0:
        pass
    else:
        return
    
    global FRAME_NB

    c = c_device.cpu()
    mass = evalTotalMass(c, config["dX"]) if eval else None
    print(f"Total mass at time {t_elapsed:08.1f}: {mass:.6f}") if eval else None
    writeVTK(c, t_elapsed, config, path=path, filename=filename, desc=desc)
    print(f"Frame {t} written in {path}{filename}_{t_elapsed:08.1f}_{FRAME_NB}.vtk.")
    
    FRAME_NB += 1 if eval else 0
    return


def plotComp(c: torch.Tensor):
    nx, ny, nz = c.shape
    freqs = np.linspace(0, nz-1, num=5, dtype=int)
    for i in freqs:
        plt.imshow(c[:, :, i], cmap='gray')
        plt.title(f"Slice {i}")
        plt.imsave(f"output/slice/slice_{i}.png", c[:, :, i], cmap='gray')

    return


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