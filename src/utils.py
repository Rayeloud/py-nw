import torch
import numpy as np
RFFT = True

def compute_fftn(input:torch.Tensor, out:torch.Tensor=None, filter:torch.Tensor=None):
    fft_dim = (0, 1, 2) if input.ndim == 3 else (0, 1)
    if out is None:
        out = torch.fft.rfftn(input, dim=fft_dim) if RFFT else torch.fft.fftn(input, dim=fft_dim)
        #out.mul_(filter) if filter is not None else None
        return out
    else:
        torch.fft.rfftn(input, out=out, dim=fft_dim) if RFFT else torch.fft.fftn(input, out=out, dim=fft_dim)
        #torch.fft.fftn(input, out=out, dim=(0, 1, 2))
        #out.mul_(filter) if filter is not None else None


def compute_ifftn(input:torch.Tensor, out:torch.Tensor=None, size:torch.Tensor=None):
    fft_dim = (0, 1, 2) if input.ndim == 3 else (0, 1)
    if out is None:
        return torch.fft.irfftn(input, s=size, dim=fft_dim) if RFFT else torch.fft.ifftn(input, dim=fft_dim).real
        #return torch.fft.ifftn(input, dim=(0, 1, 2)).real
    else:
        torch.fft.irfftn(input, s=out.size(), out=out, dim=fft_dim) if RFFT else out.copy_(torch.fft.ifftn(input, dim=fft_dim))
        #out.copy_(torch.fft.ifftn(input, dim=(0, 1, 2)).real)



def evalTotalMass(c_device: torch.Tensor, dx: float)->float:
    """
    Evaluate the mass of the composition field.

    :param c: torch.Tensor, the composition field

    :return: float, the mass of the composition field
    """
    return (torch.mean(c_device)).item()


def evalTotalEnergy(c_device: torch.Tensor, jk: "list[torch.Tensor]", params: list, filter:torch.Tensor=None)->float:
    """
    Evaluate the total energy of the composition field.

    :param c: torch.Tensor, the composition field
    :param jk: list of torch.Tensor, the wave vectors
    :param params: list of float, the physical parameters
    :param filter: torch.Tensor, the filter to apply to the Fourier transform

    :return: float, the total energy of the composition field
    """
    A, kappa, c_zero, dx = params

    # evaluate the bulk free energy (double well potential)
    c_alpha = c_zero
    c_beta = 1.0 - c_zero
    f_0 = A * ((c_device-c_alpha)**2) * ((c_beta - c_device)**2)

    c_device_fft = compute_fftn(c_device, filter=filter)

    # compute square of the gradient magnitude
    grad_c_squared = torch.zeros_like(c_device)
    for i in range(c_device.ndim):
        grad_c_squared += compute_ifftn(jk[i]*c_device_fft, size=c_device.size())**2

    # compute the interfacial energy
    f_int = kappa/2 * grad_c_squared

    N = len(c_device.view(-1))
    # compute the total energy
    total_energy = (torch.sum(f_0 + f_int)).item() * dx**c_device.ndim
    
    return total_energy


def update_geom(R1=None, R2=None, SHAPE=None, angle=None, dist=None, offset=None, dom_size=None, dx=None, file=f'../geo/nanowire_data.pro'):
    idx = {'R1': 1, 'R2': 2, 'nb': 5, 'shape': 8, 'angle': 11, 'dist': 14, 'offset': 17, 'Lx': 20, 'Ly': 21,'Lz': 22, 'dx': 23}
    with open(file, 'r') as f:
        lines = f.readlines()
        
        if R1 is not None:
            lines[idx['R1']] = f'R_1 = {R1};\n'
            lines[idx['nb']] = f'NB_NW = 1;\n'
        if R2 is not None:
            lines[idx['R2']] = f'R_2 = {R2};\n'
            lines[idx['nb']] = f'NB_NW = 2;\n'
        if SHAPE is not None:
            lines[idx['shape']] = f'SHAPE = {SHAPE};\n'
        if angle is not None:
            lines[idx['angle']] = f'angle = {angle};\n'
        if dist is not None:
            lines[idx['dist']] = f'distance = {dist};\n'
        if offset is not None:
            lines[idx['offset']] = f'offset = {offset};\n'
        if dom_size is not None:
            Lx, Ly, Lz = dom_size
            lines[idx['Lx']] = f'Lx = {Lx};\n'
            lines[idx['Ly']] = f'Ly = {Ly};\n'
            lines[idx['Lz']] = f'Lz = {Lz};\n'
        if dx is not None:
            lines[idx['dx']] = f'dX = {dx};\n'

        # write the new lines
        with open(file, 'w') as f:
            f.writelines(lines)

    return

'''
    WIP
    #f_0_0 = A*c0_device*c0_device*(1.0 - c0_device)*(1.0 - c0_device) if c0_device is not None else 0.0
    f_0_1 = A*c_device*c_device*(1.0 - c_device)*(1.0 - c_device)
    f_0_2 = A*(1-c_device-c0_device)*(1-c_device-c0_device)*(c_device+c0_device)*(c_device+c0_device) if c0_device is not None else 0.0
    #f_0 = f_0_0 + f_0_1 + f_0_2
    f_0 = f_0_1 + f_0_2

    compute_fftn(c_device, out=c_device_fft, filter=filter)
    c2_device_fft = compute_fftn(1-c_device-c0_device, filter=filter) if c0_device is not None else None

    grad_c_1 = torch.fft.ifftn(1j*kx*c_device_fft, dim=(0, 1, 2)).real
    grad_c_2 = torch.fft.ifftn(1j*ky*c_device_fft, dim=(0, 1, 2)).real
    grad_c_3 = torch.fft.ifftn(1j*kz*c_device_fft, dim=(0, 1, 2)).real

    grad_c2_1 = torch.fft.ifftn(1j*kx*c2_device_fft, dim=(0, 1, 2)).real if c0_device is not None else None
    grad_c2_2 = torch.fft.ifftn(1j*ky*c2_device_fft, dim=(0, 1, 2)).real if c0_device is not None else None
    grad_c2_3 = torch.fft.ifftn(1j*kz*c2_device_fft, dim=(0, 1, 2)).real if c0_device is not None else None
    
    grad_c_sq = grad_c_1**2 + grad_c_2**2 + grad_c_3**2
    grad_c2_sq = grad_c2_1**2 + grad_c2_2**2 + grad_c2_3**2 if c0_device is not None else 0.0

    #f_int_0 = kappa/2 * grad_c0_norm_device * grad_c0_norm_device if c0_device is not None else 0.0
    f_int_1 = kappa/2 * grad_c_sq
    f_int_2 = kappa/2 * grad_c2_sq

    #f_int = f_int_0 + f_int_1 + f_int_2
    f_int = f_int_1 + f_int_2

    N = c_device.shape[0]*c_device.shape[1]*c_device.shape[2]
    total_energy = (torch.sum(f_0 + f_int)).item()/N
'''