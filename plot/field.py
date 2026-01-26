import numpy as np

import cylinder


class DielectricField:
    """
    Electric field model for a coaxial cylindrical geometry with a dielectric layer.
    """

    eps_0 = 8.854e-12
    eps_g = eps_0
    eps_d = 5 * eps_0

    r_a = 13.5e-3
    r_d = 14.8e-3
    r_b = 18e-3

    L = 20e-2

    V_0 = 10e3

    geometric_factor = eps_g / (eps_g * np.log(r_d / r_a) + eps_d * np.log(r_b / r_d))

    coords = cylinder.CoaxialCylinder(r_a, r_b, L)

    def calculate_field(self):
        """
        Compute electric field and return Cartesian vectors.
        """
        r_safe, z = self.coords.coordinates

        r_safe = np.maximum(r_safe, 1e-15)

        half_L = self.L / 2

        term1 = (z + half_L) / np.sqrt(r_safe**2 + (z + half_L) ** 2)
        term2 = (z - half_L) / np.sqrt(r_safe**2 + (z - half_L) ** 2)

        axial_factor = (term1 - term2) / 2

        Er = self.V_0 * self.geometric_factor / r_safe * axial_factor

        Ez = (
            self.V_0
            * self.geometric_factor
            * (
                1 / np.sqrt(r_safe**2 + (z - half_L) ** 2)
                - 1 / np.sqrt(r_safe**2 + (z + half_L) ** 2)
            )
        )

        mask_diel = r_safe >= self.r_d
        factor_diel = self.eps_d / self.eps_g

        Er_diel = np.where(mask_diel, Er * factor_diel, Er)
        Ez_diel = np.where(mask_diel, Ez * factor_diel, Ez)

        return self.coords.to_cartesian(Er_diel, Ez_diel)
