"""
solver.py

This module contains functions for solving the Cahn-Hilliard equation in Fourier space.

The equation to solve is the following:

    beta c_t+1 = beta c_t + j dt k @ { (1-a*(c_t)^2) * [j k (g + 2 kappa k@k c_t)]_r }_k

where:
- beta is (1 + alpha * dt * kappa * k^4)
- a is the height of the double well potential
- g is the gradient of the bulk free energy df_0/dc
- kappa is the interfacial free energy
- k is the wave vector
- dt is the time step
- c_t is the composition field at time t
- c_t+1 is the composition field at time t+1
- @ is the dot product
- j is the imaginary unit
- []_r is the backward Fourier transform
- []_k is the forward Fourier transform
"""

import torch
import numpy as np
from tqdm import tqdm
from processing import saveFrame

def compute_g(out, c_device, A):
    """
    Compute the gradient of the bulk free energy df_0/dc

    :param out: torch.Tensor, the output tensor
    :param c_device: torch.Tensor, the composition field
    :param A: float, the height of the double well potential
    """
    factor = 2.0*A*c_device
    # 2.0*A*c_device*(1.0 - c_device)*(1.0 - 2.0*c_device)
    # => 2.0*A*c_device -6.0*A*c_device*c_device + 4.0*A*c_device*c_device*c_device
    out.copy_(factor - 3.0*factor*c_device + 2.0*factor*c_device*c_device)


def compute_vm(out, c_device, M):
    """
    Compute the variable mobility M(c) = sqrt(|c-c^2|)

    :param out: torch.Tensor, the output tensor
    :param c_device: torch.Tensor, the composition field
    """
    out.copy_(M*torch.sqrt(torch.abs(c_device - c_device*c_device)))


def compute_g_bis(out, c_device, A):
    out.copy_(-c_device + c_device*c_device*c_device)


def compute_vm_bis(out, c_device, M):
    out.copy_(1.0-c_device*c_device)


def compute_r(out, c_device_fft, g_device_fft, kappa, k2):
    """
    Compute the r=(g + kappa k@k c) term in Fourier space

    :param out: torch.Tensor, the output tensor
    :param c_device_fft: torch.Tensor, the composition field in Fourier space
    :param g_device_fft: torch.Tensor, the gradient of the bulk free energy in Fourier space
    :param kappa: float, the interfacial free energy
    :param k2: float, the wave vector squared
    """
    # r_real = g_device_fft.real + factor * c_device_fft.real
    # r_complex = g_device_fft.imag + factor * c_device_fft.imag
    # out.real.copy_(g_device_fft.real + factor * c_device_fft.real)
    # out.imag.copy_(g_device_fft.imag + factor * c_device_fft.imag)
    out.copy_(g_device_fft + kappa * k2*c_device_fft)


def compute_y(out, r_device_fft, kx, ky, kz):
    """
    Compute the j k (r) term in Fourier space (which will be brought back to real space)

    :param out: tuple of torch.Tensor, the output tensors
    :param r_device_fft: torch.Tensor, the r term in Fourier space
    :param kx: torch.Tensor, the wave vector in x direction
    :param ky: torch.Tensor, the wave vector in y direction
    :param kz: torch.Tensor, the wave vector in z direction
    """
    # r_real = r_device_fft.real
    # r_imag = r_device_fft.imag

    # out[0].real.copy_(- kx * r_imag)
    # out[0].imag.copy_(kx * r_real)
    # out[1].real.copy_(- ky * r_imag)
    # out[1].imag.copy_(ky * r_real)
    # out[2].real.copy_(- kz * r_imag)
    # out[2].imag.copy_(kz * r_real)

    out[0].copy_(torch.fft.ifftn(1j*kx*r_device_fft, dim=(0, 1, 2)))
    out[1].copy_(torch.fft.ifftn(1j*ky*r_device_fft, dim=(0, 1, 2)))
    out[2].copy_(torch.fft.ifftn(1j*kz*r_device_fft, dim=(0, 1, 2)))



def compute_q(out, y1_device, y2_device, y3_device, phi_device):
    """
    Compute the factor in real space: VM(c) * y_real where y_real = [jk * (r)]_r

    :param out: torch.Tensor, the output tensor
    :param y1_device: torch.Tensor, the y1 term in real space
    :param y2_device: torch.Tensor, the y2 term in real space
    :param y3_device: torch.Tensor, the y3 term in real space
    :param phi_device: torch.Tensor, the VM(c) term in real space
    :param M: torch.Tensor, the VM(c) term in real space
    """
    # out[0].copy_(phi_device*y1_device)
    # out[1].copy_(phi_device*y2_device)
    # out[2].copy_(phi_device*y3_device)
    torch.fft.fftn(phi_device*y1_device, dim=(0, 1, 2), out=out[0])
    torch.fft.fftn(phi_device*y2_device, dim=(0, 1, 2), out=out[1])
    torch.fft.fftn(phi_device*y3_device, dim=(0, 1, 2), out=out[2])


def compute_euler_timestep(out, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k4, kappa, alpha, dt):
    """
    Solve the Cahn-Hilliard equation in Fourier space

    :param out: torch.Tensor, the output tensor
    :param q1_device_fft: torch.Tensor, the q1 term in Fourier space
    :param q2_device_fft: torch.Tensor, the q2 term in Fourier space
    :param q3_device_fft: torch.Tensor, the q3 term in Fourier space
    :param kx: torch.Tensor, the wave vector in x direction
    :param ky: torch.Tensor, the wave vector in y direction
    :param kz: torch.Tensor, the wave vector in z direction
    :param k4: torch.Tensor, the square of the wave vector squared
    :param kappa: float, the interfacial free energy
    :param alpha: float, the stability parameter
    :param dt: float, the time step
    """
    denominator = 1.0 + alpha * dt * kappa * k4
    # out.real.add_(-dt*(kx*q1_device_fft.imag +
    #                    ky*q2_device_fft.imag +
    #                    kz*q3_device_fft.imag).div_(denominator))
    # out.imag.add_(dt*(kx*q1_device_fft.real +
    #                   ky*q2_device_fft.real +
    #                   kz*q3_device_fft.real).div_(denominator))
    out.add_(dt*1j*(kx*q1_device_fft+ky*q2_device_fft+kz*q3_device_fft).div_(denominator))
    

@torch.no_grad
def solve_cahn_hilliard(c, config, T, param, mode, path, filename):
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

    dx, dy, dz = config["dX"], config["dX"], config["dX"]
    save_every, dt = param['save_every'], param['dt']

    t_start, t_end = T
    n_steps = int((t_end-t_start)/dt)

    # define fourier space
    kx = torch.fft.fftfreq(c.shape[0], d=dx) * 2.0 * np.pi
    ky = torch.fft.fftfreq(c.shape[1], d=dy) * 2.0 * np.pi
    kz = torch.fft.fftfreq(c.shape[2], d=dz) * 2.0 * np.pi

    kx_mesh, ky_mesh, kz_mesh = torch.meshgrid(kx, ky, kz, indexing='ij')

    k2 = kx_mesh**2 + ky_mesh**2 + kz_mesh**2
    k4 = k2*k2
    # init auxilary tensors
    # bring to device
    kx_mesh = kx_mesh.to(device=device)
    ky_mesh = ky_mesh.to(device=device)
    kz_mesh = kz_mesh.to(device=device)

    k2 = k2.to(device=device)
    k4 = k4.to(device=device)

    c_device = c.to(device=device)

    # preallocate tensors in device before the loop
    torch.set_default_device(device=device)

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
    rhs_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
    
    # start loop
    for t in tqdm(range(t_start+1, t_start+n_steps+1), desc=f'Solving...'):
        # time stepping function (either semi-implicit euler or RK4)
        compute_timestep(c_device, c_device_fft, g_device, g_device_fft, phi_device, r_device_fft, 
                y1_device, y2_device, y3_device, q1_device_fft, q2_device_fft, q3_device_fft, rhs_device_fft, 
                kx_mesh, ky_mesh, kz_mesh, k2, k4, param, out=c_device_fft, mode=mode)
        
        c_device.copy_(torch.fft.ifftn(c_device_fft, dim=(0, 1, 2), out=c_device_fft))
        saveFrame(c_device, t, save_every, config, path=path, filename=filename)
    
    return

def compute_timestep(c_device, c_device_fft, g_device, g_device_fft, phi_device, r_device_fft, 
                y1_device, y2_device, y3_device, q1_device_fft, q2_device_fft, q3_device_fft, rhs_device_fft, 
                kx, ky, kz, k2, k4, param, out, mode='Euler'):
    if mode=='Euler':
        alpha, kappa, dt = param['alpha'], param['kappa'], param['dt']
        denominator = 1 + alpha * dt * kappa * k4
        compute_rhs(rhs_device_fft, c_device, c_device_fft, g_device, g_device_fft, phi_device, r_device_fft, 
                y1_device, y2_device, y3_device, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k2, param)
        out.copy_(c_device_fft + dt*rhs_device_fft.div_(denominator))
    elif mode=='RK4':
        alpha, kappa, dt = param['alpha'], param['kappa'], param['dt']
        compute_rhs(rhs_device_fft, c_device, c_device_fft, g_device, g_device_fft, phi_device, r_device_fft, 
                y1_device, y2_device, y3_device, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k2, param)
        c_1_fft = c_device_fft + dt * rhs_device_fft
        c_1 = torch.fft.ifftn(c_1_fft, dim=(0, 1, 2))
        compute_rhs(rhs_device_fft, c_1, c_1_fft, g_device, g_device_fft, phi_device, r_device_fft, 
                y1_device, y2_device, y3_device, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k2, param)
        c_2_fft = 3/4*c_device_fft + 1/4*c_1_fft + dt/4*rhs_device_fft
        c_2 = torch.fft.ifftn(c_2_fft, dim=(0, 1, 2))
        compute_rhs(rhs_device_fft, c_2, c_2_fft, g_device, g_device_fft, phi_device, r_device_fft, 
                y1_device, y2_device, y3_device, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k2, param)
        out.copy_(1/3*c_device_fft+2/3*c_2_fft+2*dt/3*rhs_device_fft)

def compute_rhs(out, c_device, c_device_fft, g_device, g_device_fft, phi_device, r_device_fft, 
                y1_device, y2_device, y3_device, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k2, param):
    M, A, kappa  = param['M'], param['A'], param['kappa']
    compute_vm(phi_device, c_device, M)

    compute_g(g_device, c_device, A)

    g_device_fft.copy_(g_device)
    c_device_fft.copy_(c_device)

    torch.fft.fftn(g_device_fft, dim=(0, 1, 2), out=g_device_fft)
    torch.fft.fftn(c_device_fft, dim=(0, 1, 2), out=c_device_fft)

    # compute r
    compute_r(r_device_fft, c_device_fft, g_device_fft, kappa, k2)

    # compute y
    compute_y((y1_device, y2_device, y3_device), r_device_fft, kx, ky, kz)

    # compute q
    compute_q((q1_device_fft, q2_device_fft, q3_device_fft), y1_device, y2_device, y3_device, phi_device)

    out.copy_(1j*(kx*q1_device_fft + ky*q2_device_fft + kz*q3_device_fft))