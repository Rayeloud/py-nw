import torch
import numpy as np
RFFT = True


def compute_fftn(input:torch.Tensor, out:torch.Tensor=None, filter:torch.Tensor=None):
    fft_dim3 = (0, 1, 2)
    fft_dim2 = (0, 1)
    if RFFT:
        if out is None:
            out = torch.fft.rfftn(input, dim=fft_dim3) if input.ndim == 3 else torch.fft.rfftn(input, dim=fft_dim2)
            out.mul_(filter) if filter is not None else None
            return out
        else:
            torch.fft.rfftn(input, out=out, dim=fft_dim3) if input.ndim == 3 else torch.fft.rfftn(input, out=out, dim=fft_dim2)
            #torch.fft.fftn(input, out=out, dim=(0, 1, 2))
            out.mul_(filter) if filter is not None else None
    else:
        if out is None:
            out = torch.fft.fftn(input, dim=fft_dim3) if input.ndim == 3 else torch.fft.fftn(input, dim=fft_dim2)
            out.mul_(filter) if filter is not None else None
            return out
        else:
            torch.fft.fftn(input, out=out, dim=fft_dim3) if input.ndim == 3 else torch.fft.fftn(input, out=out, dim=fft_dim2)
            #torch.fft.fftn(input, out=out, dim=(0, 1, 2))
            out.mul_(filter) if filter is not None else None


def compute_ifftn(input:torch.Tensor, out:torch.Tensor=None, size:torch.Tensor=None):
    fft_dim3 = (0, 1, 2)
    fft_dim2 = (0, 1)
    if RFFT:
        if out is None:
            return torch.fft.irfftn(input, s=size, dim=fft_dim3) if input.ndim == 3 else torch.fft.irfftn(input, s=size, dim=fft_dim2) #if RFFT else torch.fft.ifftn(input, dim=fft_dim).real
            #return torch.fft.ifftn(input, dim=(0, 1, 2)).real
        else:
            torch.fft.irfftn(input, s=out.size(), out=out, dim=fft_dim3) if input.ndim == 3 else torch.fft.irfftn(input, s=out.size(), out=out, dim=fft_dim2) #if RFFT else out.copy_(torch.fft.ifftn(input, dim=fft_dim))
            #out.copy_(torch.fft.ifftn(input, dim=(0, 1, 2)).real)
    else:
        if out is None:
            return torch.fft.ifftn(input, dim=fft_dim3).real if input.ndim == 3 else torch.fft.ifftn(input, dim=fft_dim2).real #if RFFT else torch.fft.ifftn(input, dim=fft_dim).real
            #return torch.fft.ifftn(input, dim=(0, 1, 2)).real
        else:
            out.copy_(torch.fft.ifftn(input, dim=fft_dim3).real) if input.ndim == 3 else out.copy_(torch.fft.ifftn(input, dim=fft_dim2).real) #if RFFT else out.copy_(torch.fft.ifftn(input, dim=fft_dim))
            #out.copy_(torch.fft.ifftn(input, dim=(0, 1, 2)).real)



def evalTotalMass(c_device: torch.Tensor, dx: float)->float:
    """
    Evaluate the mass of the composition field.

    :param c: torch.Tensor, the composition field

    :return: float, the mass of the composition field
    """
    # return (torch.mean(c_device)).item()
    return (torch.sum(c_device)).item()


def evalTotalEnergy(c_device: torch.Tensor, c_device_fft: torch.Tensor, jk: "list[torch.Tensor]", params: list, filter:torch.Tensor=None)->float:
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

def evalCosThetaB(c_device: torch.Tensor, c_device_fft: torch.Tensor, grad_wall_device: "list[torch.Tensor]", norm_grad_wall_device, jk: "list[torch.Tensor]", params: list, filter: torch.Tensor=None)->torch.Tensor:
    kappa, A = params
    # grad wall x grad c
    grad_wall_x_grad_c = compute_ifftn(jk[0]*c_device_fft, size=c_device.size()) * grad_wall_device[0] + compute_ifftn(jk[1]*c_device_fft, size=c_device.size())*grad_wall_device[1]
    if c_device.ndim == 3:
        grad_wall_x_grad_c += compute_ifftn(jk[2]*c_device_fft, size=c_device.size())*grad_wall_device[2]

    # norm grad c x norm grad wall
    norm_grad_c_x_norm_grad_wall = np.sqrt(2.0/kappa * A)* torch.abs(c_device*(1.0 - c_device)) * norm_grad_wall_device + 1e-5 # to avoid division by zero

    return grad_wall_x_grad_c.div_(norm_grad_c_x_norm_grad_wall)


def evalStructureFactor(c_device: torch.Tensor, k: "list[torch.Tensor]", filter:torch.Tensor=None)->torch.Tensor:
    """
    Evaluate the structure factor of the composition field.

    :param c: torch.Tensor, the composition field
    :param jk: list of torch.Tensor, the wave vectors
    :param params: list of float, the physical parameters
    :param filter: torch.Tensor, the filter to apply to the Fourier transform

    :return: torch.Tensor, the structure factor of the composition field
    """
    mean_c_sq = torch.mean(c_device)
    mean_sq_c = torch.mean(c_device**2)
    c_device_fft = compute_fftn(c_device - mean_c_sq, out=None, filter=filter)
    mean_c_sq = mean_c_sq**2

    SF = torch.mean(c_device_fft*c_device_fft.conj()).real
    SF = SF/(mean_sq_c - mean_c_sq)

    return SF

# Function to update the image
def update_plot(fig, ax, im, update_data, t):
    im.set_data(update_data)  # Update the data of the imshow
    ax.set_title(f"$t={t}$")  # Optional: update title or other properties
    fig.canvas.draw()  # Redraw the figure
    fig.canvas.flush_events()  # Flush GUI events


def update_geom(R1=None, R2=None, SHAPE=None, SHAPE_EQL=None, SUBSTRATE=None, angle=None, dist=None, offset=None, dom_size=None, dx=None, file=f'../geo/nanowire_data.pro'):
    idx = {'R1': 1, 'R2': 2, 'nb': 5, 'shape': 8, 'shape_eql': 9, 'substrate':10, 'angle': 13, 'dist': 16, 'offset': 19, 'Lx': 22, 'Ly': 23,'Lz': 24, 'dx': 25}
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
        if SHAPE_EQL is not None:
            lines[idx['shape_eql']] = f'SHAPE_EQL = {SHAPE_EQL};\n'
        if SUBSTRATE is not None:
            lines[idx['substrate']] = f'SUBSTRATE = {SUBSTRATE};\n'
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