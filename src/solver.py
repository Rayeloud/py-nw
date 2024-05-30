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
    Compute the square root of the absolute value of the composition field

    :param out: torch.Tensor, the output tensor
    :param c_device: torch.Tensor, the composition field
    """
    out.copy_(M*torch.sqrt(torch.abs(c_device - c_device*c_device)))


def compute_r(out, c_device_fft, g_device_fft, kappa, k2):
    """
    Compute the r=(g + 2 kappa k@k c) term in Fourier space (kappa = kappa/2 here)

    :param out: torch.Tensor, the output tensor
    :param c_device_fft: torch.Tensor, the composition field in Fourier space
    :param g_device_fft: torch.Tensor, the gradient of the bulk free energy in Fourier space
    :param kappa: float, the interfacial free energy
    :param k2: float, the wave vector squared
    """
    factor = kappa * k2
    # r_real = g_device_fft.real + factor * c_device_fft.real
    # r_complex = g_device_fft.imag + factor * c_device_fft.imag
    out.real.copy_(g_device_fft.real + factor * c_device_fft.real)
    out.imag.copy_(g_device_fft.imag + factor * c_device_fft.imag)


def compute_y(out, r_device_fft, kx, ky, kz):
    """
    Compute the j k (r) term in Fourier space (which will be brought back to real space)

    :param out: tuple of torch.Tensor, the output tensors
    :param r_device_fft: torch.Tensor, the r term in Fourier space
    :param kx: torch.Tensor, the wave vector in x direction
    :param ky: torch.Tensor, the wave vector in y direction
    :param kz: torch.Tensor, the wave vector in z direction
    """
    r_real = r_device_fft.real
    r_imag = r_device_fft.imag

    # y1_real = - kx * r_imag
    # y1_imag = kx * r_real
    # y2_real = - ky * r_imag
    # y2_imag = ky * r_real
    # y3_real = - kz * r_imag
    # y3_imag = kz * r_real

    # out[0].copy_(torch.complex(y1_real, y1_imag))
    # out[1].copy_(torch.complex(y2_real, y2_imag))
    # out[2].copy_(torch.complex(y3_real, y3_imag))

    out[0].real.copy_(- kx * r_imag)
    out[0].imag.copy_(kx * r_real)
    out[1].real.copy_(- ky * r_imag)
    out[1].imag.copy_(ky * r_real)
    out[2].real.copy_(- kz * r_imag)
    out[2].imag.copy_(kz * r_real)


def compute_q(out, y1_device, y2_device, y3_device, phi_device):
    """
    Compute the factor in real space: VM(c) * y_real where y_real = [j k (r)]_r

    :param out: torch.Tensor, the output tensor
    :param y1_device: torch.Tensor, the y1 term in real space
    :param y2_device: torch.Tensor, the y2 term in real space
    :param y3_device: torch.Tensor, the y3 term in real space
    :param phi_device: torch.Tensor, the VM(c) term in real space
    :param M: torch.Tensor, the VM(c) term in real space
    """
    # to take into account the bulk mobility contribution
    # (if M=0, then the mobility is constant and equal to 1)
    #factor = phi_device # + (1.0 - np.ceil(M))
    out[0].copy_(phi_device*y1_device)
    out[1].copy_(phi_device*y2_device)
    out[2].copy_(phi_device*y3_device)


def solve_ch(out, q1_device_fft, q2_device_fft, q3_device_fft, kx, ky, kz, k4, kappa, alpha, dt):
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
    out.real.add_(-dt*(kx*q1_device_fft.imag +
                       ky*q2_device_fft.imag +
                       kz*q3_device_fft.imag).div_(denominator))
    out.imag.add_(dt*(kx*q1_device_fft.real +
                      ky*q2_device_fft.real +
                      kz*q3_device_fft.real).div_(denominator))
