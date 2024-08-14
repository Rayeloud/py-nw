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

from utils import compute_fftn, compute_ifftn

def compute_g(out, c_device, A, c_zero=1.0):
    """
    Compute the gradient of the bulk free energy df_0/dc

    :param out: torch.Tensor, the output tensor
    :param c_device: torch.Tensor, the composition field
    :param A: float, the height of the double well potential
    """
    #out.copy_(2.0*A*(c_device - c_eq[1])*(c_eq[0] - c_device)*(c_eq[0] + c_eq[1] - 2.0*c_device))
    out.copy_(2.0*A*(c_device-(1-c_zero))*(c_zero - c_device)*(1 - 2.0*c_device))


def compute_vm(out, c_device, M, a=1.0):
    """
    Compute the variable mobility M(c) = sqrt(|c-c^2|)

    :param out: torch.Tensor, the output tensor
    :param c_device: torch.Tensor, the composition field
    """
    #out.copy_(M*torch.abs(1-c_device*c_device))
    out.copy_(float(M*(1.0-a)) + 2*a*M*torch.sqrt(torch.abs(c_device - c_device*c_device)))
    # c_alpha = 1.0
    # c_beta = 0.0
    # out.copy_(M*(1-a) + a*M*torch.sqrt((c_device - c_alpha)*(c_device - c_alpha)*(c_beta - c_device)*(c_beta - c_device)))

    #out.copy_(8.0 * M * c_device * c_device * (1.0 - c_device) * (1.0 - c_device))
    #out.copy_(2.0 * M * c_device * (1.0 - c_device))


#def compute_mu_fft(out, c_device_fft, g_device_fft, kappa, k2):
def compute_mu_fft(out, c_device_fft, g_device_fft, kappa, k2):
    """
    Compute the mu_hat=(g + kappa k@k c) term in Fourier space (chemical potential in Fourier space)

    :param out: torch.Tensor, the output tensor
    :param c_device_fft: torch.Tensor, the composition field in Fourier space
    :param g_device_fft: torch.Tensor, the gradient of the bulk free energy in Fourier space
    :param kappa: float, the interfacial free energy
    :param k2: float, the wave vector squared
    """
    out.copy_(g_device_fft + kappa * k2 * c_device_fft)


def compute_mu_fft_SBM(out, c_device, c_device_fft, g_device_fft, wall_device_fft, grad_wall_device, norm_grad_c_wall_device, params, jk, k2, filter=None):
    A, kappa = params
    term1 = compute_fftn(compute_ifftn(jk[0]*c_device_fft).mul_(grad_wall_device[0])
                        + compute_ifftn(jk[1]*c_device_fft).mul_(grad_wall_device[1])
                        + compute_ifftn(jk[2]*c_device_fft).mul_(grad_wall_device[2]), filter=filter)
    term2 = torch.sqrt(2*A*(c_device-1)**2 * c_device**2)

    out.copy_(g_device_fft )


def compute_mu_fft_bis(out, c_device, c_device_fft, 
                       flux_c_device_fft, bc_device_fft, g_device_fft, 
                       grad_indic_device, grad_indic_norm_device, low_pass_filter,
                       kappa, k, k2, theta):
    # contact angle=pi/2
    grad_c1_device = torch.fft.ifftn(1j*k[0]*c_device_fft, dim=(0, 1, 2)).real
    grad_c2_device = torch.fft.ifftn(1j*k[1]*c_device_fft, dim=(0, 1, 2)).real
    grad_c3_device = torch.fft.ifftn(1j*k[2]*c_device_fft, dim=(0, 1, 2)).real

    compute_fftn(grad_c1_device*grad_indic_device[0] + grad_c2_device*grad_indic_device[1] + grad_c3_device*grad_indic_device[2], out=flux_c_device_fft, filter=low_pass_filter)
    compute_fftn(c_device*(1-c_device)/(np.sqrt(2*kappa)) * grad_indic_norm_device * np.cos(theta), out=bc_device_fft, filter=low_pass_filter)

    out.copy_(g_device_fft + kappa * k2*c_device_fft + bc_device_fft + flux_c_device_fft)


#def compute_grad_mu(out:"list[torch.Tensor]", mu_device_fft:torch.Tensor, k:"list[torch.Tensor]"):
def compute_grad_mu(out, mu_device_fft, k):
    """
    Compute the j k (mu_hat) term in Fourier space (which will be brought back to real space)
    Gradient of the chemical potential in Fourier space

    :param out: list of torch.Tensor, the output tensors
    :param r_device_fft: torch.Tensor, the r term in Fourier space
    :param k: list[torch.Tensor], the wave vector in x direction
    """
    # out[0].copy_(torch.fft.ifftn(jkx*mu_device_fft, dim=(0, 1, 2)))
    # out[1].copy_(torch.fft.ifftn(jky*mu_device_fft, dim=(0, 1, 2)))
    # out[2].copy_(torch.fft.ifftn(jkz*mu_device_fft, dim=(0, 1, 2)))
    # compute_ifftn(jk[0]*mu_device_fft, out=out[0])
    # compute_ifftn(jk[1]*mu_device_fft, out=out[1])
    # compute_ifftn(jk[2]*mu_device_fft, out=out[2])
    if len(out) > 2:
        compute_ifftn(1j*k[0]*mu_device_fft, out=out[0])
        compute_ifftn(1j*k[1]*mu_device_fft, out=out[1])
        compute_ifftn(1j*k[2]*mu_device_fft, out=out[2])
    else:
        compute_ifftn(1j*k[0]*mu_device_fft, out=out[0])
        compute_ifftn(1j*k[1]*mu_device_fft, out=out[1])


#def compute_flux_fft(out:"list[torch.Tensor]", grad_mu_device:"list[torch.Tensor]", phi_device:torch.Tensor, low_pass_filter=None):
def compute_flux_fft(out, grad_mu_device, phi_device, low_pass_filter=None):
    """
    Compute the factor in real space: VM(c) * grad mu where grad mu = [jk * mu_hat]_r

    :param out: torch.Tensor, the output tensor
    :param grad_mu_device: torch.Tensor, the gradient of the chemical potential in real space
    :param phi_device: torch.Tensor, the VM(c) term in real space
    :param M: torch.Tensor, the VM(c) term in real space
    """
    #torch.fft.rfftn(phi_device*grad_mu1_device, dim=(0, 1, 2), out=out[0])
    #torch.fft.rfftn(phi_device*grad_mu2_device, dim=(0, 1, 2), out=out[1])
    #torch.fft.rfftn(phi_device*grad_mu3_device, dim=(0, 1, 2), out=out[2])
    # compute_fftn(phi_device*grad_mu_device[0], out=out[0], filter=low_pass_filter)
    # compute_fftn(phi_device*grad_mu_device[1], out=out[1], filter=low_pass_filter)
    # compute_fftn(phi_device*grad_mu_device[2], out=out[2], filter=low_pass_filter)
    if len(out) > 2:
        compute_fftn(phi_device*grad_mu_device[0], out=out[0], filter=low_pass_filter)
        compute_fftn(phi_device*grad_mu_device[1], out=out[1], filter=low_pass_filter)
        compute_fftn(phi_device*grad_mu_device[2], out=out[2], filter=low_pass_filter)
    else:
        compute_fftn(phi_device*grad_mu_device[0], out=out[0], filter=low_pass_filter)
        compute_fftn(phi_device*grad_mu_device[1], out=out[1], filter=low_pass_filter)


#def compute_euler_timestep(out:torch.Tensor, flux_device_fft:"list[torch.Tensor]", jk:"list[torch.Tensor]", k4:torch.Tensor, params:"list[float]"):
def compute_euler_timestep(out, flux_device_fft, k, k4, params):
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
    kappa, alpha, dt = params
    denominator = 1.0 + alpha * dt * kappa * k4
    if out.ndim > 2:
        out.add_(1j*dt*(k[0]*flux_device_fft[0] + k[1]*flux_device_fft[1] + k[2]*flux_device_fft[2]).div_(denominator))
    else:
        out.add_(1j*dt*(k[0]*flux_device_fft[0] + k[1]*flux_device_fft[1]).div_(denominator))
    

'''
def compute_rhs(out, c_device, c_device_fft, g_device, g_device_fft, phi_device, 
                mu_device_fft, grad_mu_device, flux_device_fft, k, k2, const):
    """
    Compute the right hand side of the Cahn-Hilliard equation in Fourier space

    :param out: torch.Tensor, the output tensor
    :param c_device_fft: torch.Tensor, the composition field in Fourier space
    :param g_device_fft: torch.Tensor, the gradient of the bulk free energy in Fourier space
    :param grad_mu_device: tuple of torch.Tensor, the gradient of the chemical potential in Fourier space
    :param flux_device_fft: torch.Tensor, the flux term in Fourier space
    :param k: torch.Tensor, the wave vector
    :param k2: torch.Tensor, the wave vector squared
    :param k4: torch.Tensor, the square of the wave vector squared
    :param const: float, the constant term
    """
    A, M, kappa, alpha = const
    # df/dc|eq = 2 A c (1-c) (1-2c)
    compute_g(g_device, c_device, A)

    # M(c) = m sqrt(|c-c^2|)
    compute_vm(phi_device, c_device, M)

    torch.fft.rfftn(c_device, dim=(0, 1, 2), out=c_device_fft)
    torch.fft.rfftn(g_device, dim=(0, 1, 2), out=g_device_fft)

    # mu_fft = g_fft + k^2 kappa c_fft
    compute_mu_fft(mu_device_fft, c_device_fft, g_device_fft, kappa, k2)

    # grad mu = inv_fft(jk mu_fft)
    compute_grad_mu(grad_mu_device, mu_device_fft, k[0], k[1], k[2])

    # J_fft = fft(M(c) grad mu)
    compute_flux(flux_device_fft, grad_mu_device[0], grad_mu_device[1], grad_mu_device[2], phi_device)

    # rhs = jk @ J_fft = jk @ {(M(c) [jk mu]_r}_k
    out.copy_(1j*(k[0]*flux_device_fft[0] + k[1]*flux_device_fft[1] + k[2]*flux_device_fft[2]))


def solveCH(c, t_range, const, config,device):
    t_start, t_end = t_range
    nb_steps = (t_end - t_start)/dt
    # define fourier space

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

    # preallocate tensors in device before the loop
    torch.set_default_device(device)
    # real space
    g_device = torch.zeros_like(c_device)
    phi_device = torch.zeros_like(c_device)
    # grad_mu_device
    grad_mu1_device = torch.zeros_like(c_device)
    grad_mu2_device = torch.zeros_like(c_device)
    grad_mu3_device = torch.zeros_like(c_device)

    # fourier space
    c_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
    g_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
    mu_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
    # flux_device_fft
    flux1_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
    flux2_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
    flux3_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)

    if scheme == 'euler':
        rhs_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        for t in tqdm(range(t_start+1, t_start+nb_steps+1), desc="Solving..."):
            # compute right hand side
            compute_rhs(rhs_device_fft, c_device, c_device_fft, g_device, g_device_fft, phi_device, mu_device_fft, 
                        (grad_mu1_device, grad_mu2_device, grad_mu3_device), (flux1_device_fft, flux2_device_fft, flux3_device_fft), 
                        (kx, ky, kz), k2, const)
            
            # semi-implicit euler step
            denominator = 1.0 + alpha * dt * kappa * k4

            c_device_fft.add_(dt*rhs_device_fft).div_(denominator)

            # bring back to real space
            c_device.copy_(torch.fft.ifftn(c_device_fft, dim=(0, 1, 2), out=c_device_fft))

            if torch.max(c_device).item() < 0.5:
                print(f'Warning! Order parameter c stricly below critical value.')

            saveFrame(c_device, t, save_every, config, dt, path=PATH, filename=FILENAME)
    elif scheme == 'crank-nicolson':
        c_guess_device = torch.zeros_like(c_device)
        c_guess_device_fft = torch.zeros_like(c_device_fft)
        c_new_device = torch.zeros_like(c_device)
        rhs_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        rhs_guess_device_fft = torch.zeros_like(c_device, dtype=torch.complex64)
        for t in tqdm(range(t_start+1, t_start+nb_steps+1), desc="Solving..."):
            # non-linear term
            for n in range(0, 8):
                # compute right hand side
                compute_rhs(rhs_device_fft, c_device, c_device_fft, g_device, g_device_fft, phi_device, mu_device_fft, 
                            (grad_mu1_device, grad_mu2_device, grad_mu3_device), (flux1_device_fft, flux2_device_fft, flux3_device_fft), 
                            (kx, ky, kz), k2, const)
                
                c_new

                if torch.max(c_device).item() < 0.5:
                    print(f'Warning! Order parameter c stricly below critical value.')

    

    return
'''