"""
Author: oQuintino
Description:
    Geometry and coordinate utilities for coaxial cylindrical simulations.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray


@dataclass
class CoaxialCylinder:
    """
    Representation of a coaxial cylindrical geometry with discretized coordinates.

    This class generates spatial coordinates for a hollow cylinder defined
    by inner radius, outer radius, and length. It provides utilities to:
    - Create a structured cylindrical grid
    - Convert cylindrical field components to Cartesian vectors
    - Access flattened coordinate arrays for numerical computations

    Parameters
    ----------
    r_i : float
        Inner radius of the cylinder.
    r_o : float
        Outer radius of the cylinder.
    L : float
        Total length of the cylinder along the z-axis.
    Δ : float, optional
        Spatial discretization step (default is 1e-2).
    """

    r_i: float
    r_o: float
    L: float
    Δ: float = 1e-2  # Spatial resolution

    @cached_property
    def spaced_coordinates(self):
        """
        Generate uniformly spaced cylindrical coordinates.

        The number of points in each direction is estimated from the
        geometry and spacing, with minimum thresholds to ensure
        numerical stability.

        Returns
        -------
        r : ndarray
            Radial coordinates from r_i to r_o.
        theta : ndarray
            Angular coordinates from 0 to 2π (excluded).
        z : ndarray
            Axial coordinates centered at z = 0.
        """

        # Estimated number of samples
        N_r = int((self.r_o - self.r_i) / self.Δ)
        N_theta = int(np.pi * (self.r_i + self.r_o) / self.Δ)
        N_z = int(self.L / self.Δ)

        # Enforce minimum resolution
        N_r_min = max(N_r, 10)
        N_theta_min = max(N_theta, 20)
        N_z_min = max(N_z, 5)

        r = np.linspace(self.r_i, self.r_o, N_r_min)
        theta = np.linspace(0, 2 * np.pi, N_theta_min, endpoint=False)

        half_L = self.L / 2
        z = np.linspace(-half_L, half_L, N_z_min)

        return r, theta, z

    @cached_property
    def points(self):
        """
        Generate flattened Cartesian coordinates of the cylindrical grid.

        The cylindrical grid is converted to Cartesian coordinates and
        flattened into 1D arrays suitable for vectorized numerical
        computations.

        Returns
        -------
        x_f, y_f, z_f : ndarray
            Flattened Cartesian coordinates of all grid points.
        """

        r, theta, z = self.spaced_coordinates
        rr, tt, zz = np.meshgrid(r, theta, z, indexing="ij")

        x = rr * np.cos(tt)
        y = rr * np.sin(tt)

        # Flatten arrays
        x_f = x.ravel()
        y_f = y.ravel()
        z_f = zz.ravel()

        return x_f, y_f, z_f

    @cached_property
    def rz_coordinates(self):
        """
        Return cylindrical (r, z) coordinates for all grid points.

        Useful for problems with axial symmetry where angular dependence
        is not required.

        Returns
        -------
        r : ndarray
            Radial coordinates.
        z : ndarray
            Axial coordinates.
        """

        x, y, z = self.points

        r: NDArray[float64] = np.sqrt(x**2 + y**2)

        return r, z

    def to_cartesian(self, Er: NDArray[float64], Ez: NDArray[float64]):
        """
        Convert cylindrical field components to Cartesian vectors.

        Given radial (Er) and axial (Ez) field components defined at
        the cylinder grid points, this method computes the corresponding
        Cartesian vectors and their magnitudes.

        Parameters
        ----------
        Er : ndarray of shape (N,)
            Radial field component at each grid point.
        Ez : ndarray of shape (N,)
            Axial field component at each grid point.

        Returns
        -------
        points : ndarray of shape (N, 3)
            Cartesian coordinates of the grid points.
        vectors_unit : ndarray of shape (N, 3)
            Unit Cartesian field vectors.
        mag : ndarray of shape (N,)
            Magnitude of the field vectors.
        """

        if Er.shape != Ez.shape:
            raise ValueError("Er and Ez must match the number of grid points.")

        x, y, z = self.points
        r = np.sqrt(x**2 + y**2)

        # Avoid division by zero at the axis
        r_safe = np.maximum(r, 1e-15)

        Ex = Er * (x / r_safe)
        Ey = Er * (y / r_safe)

        vectors = np.vstack((Ex, Ey, Ez)).T
        points = np.vstack((x, y, z)).T

        mag: NDArray[float64] = np.linalg.norm(vectors, axis=1)
        mag_safe: NDArray[float64] = np.maximum(mag, 1e-15)

        vectors_unit = vectors / mag_safe[:, None]

        return points, vectors_unit, mag
