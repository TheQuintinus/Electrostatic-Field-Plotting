import math

import numpy as np
import pyvista as pv

MATRICULA = 202210504463

# ------------------------------------------------------------
# Parâmetros físicos
# ------------------------------------------------------------
eps_0 = 8.854e-12
eps_g = eps_0
eps_d = 5 * eps_0

r_a = 13.5e-3
r_d = 14.8e-3
r_b = 18e-3

L = 20e-2

V_0 = 10e3

geometric_factor = eps_g / (
    eps_g * math.log(r_d / r_a)
    +
    eps_d * math.log(r_b / r_d)
)

# ------------------------------------------------------------
# Geometria do cilindro
# ------------------------------------------------------------
Δ = 1e-2   # espaçamento desejado entre vetores

N_r     = int((r_b - r_a) / Δ)
N_theta = int(2*np.pi * ((r_a+r_b)/2) / Δ)
N_z     = int(L / Δ)

# garantir mínimo
N_r     = max(N_r, 5)
N_theta = max(N_theta, 20)
N_z     = max(N_z, 5)

r     = np.linspace(r_a, r_b, N_r)
theta = np.linspace(0, 2*np.pi, N_theta)
z     = np.linspace(-L/2, L/2, N_z)

rr, tt, zz = np.meshgrid(r, theta, z, indexing="ij")
x = rr * np.cos(tt)
y = rr * np.sin(tt)
z = zz

x = x.flatten()
y = y.flatten()
z = z.flatten()

rv = np.sqrt(x*x + y*y)
mask = (rv >= r_a) & (rv <= r_b)

x = x[mask]
y = y[mask]
z = z[mask]

# ------------------------------------------------------------
# Cálculo de E_r(r,z) e E_z(r, z)
# ------------------------------------------------------------
r = np.sqrt(x*x + y*y)

term1 = (z + L/2) / np.sqrt(r**2 + (z + L/2)**2)
term2 = (z - L/2) / np.sqrt(r**2 + (z - L/2)**2)

axial_correction_factor = (term1 - term2) / 2

Er = V_0 * geometric_factor / r * axial_correction_factor

Ez = V_0 * geometric_factor * (
    1 / np.sqrt(r**2 + (z - L/2)**2)
    -
    1 / np.sqrt(r**2 + (z + L/2)**2)
)

# Vetor radial
Ex = Er * (x / r)
Ey = Er * (y / r)

# Masks for the dielectrics
mask_ard = (r >= r_a) & (r <= r_d)
mask_rdb = (r >= r_d) & (r <= r_b)

vectors = np.vstack((Ex, Ey, Ez)).T
points = np.vstack((x, y, z)).T

# ------------------------------------------------------------
# PREVINIR FLECHAS GIGANTES:
# Normalizar vetores e mostrar magnitude apenas na cor
# ------------------------------------------------------------
mag = np.linalg.norm(vectors, axis=1)
mag[mag == 0] = 1e-12
vectors_unit = vectors / mag[:,None]   # << DIREÇÃO SOMENTE <<

# ------------------------------------------------------------
# Plot com tamanho de seta constante
# ------------------------------------------------------------
pv.close_all()
plotter = pv.Plotter()
plotter.clear()

cloud = pv.PolyData(points)
cloud["vectors"] = vectors_unit
cloud["mag"] = mag                  # coloração

glyphs = cloud.glyph(
    orient="vectors",
    scale=False,     # <<--- TAMANHO FIXO
    factor=Δ * 0.2,       # <<--- AJUSTE O TAMANHO AQUI
)

plotter.add_mesh(glyphs, scalars="mag", cmap="viridis")

# Cilindro transparente
cyl = pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=r_b, height=L)
plotter.add_mesh(cyl, color="blue", opacity=0.1)

plotter.add_axes()
plotter.show()
