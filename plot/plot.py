"""
Author: oQuintino
Description:
    Visualization utilities for rendering the electric field of the coaxial
    dielectric cylinder using PyVista.
"""

import numpy as np
import pyvista as pv

from field import DielectricField, Region


class Plot:
    """
    PyVista plot of the electric field using fixed-size glyphs and magnitude-based coloring.
    """

    LINE_WIDTH = 1

    def __init__(self, glyph_size=1e-3):
        """
        Initialize the PyVista plotter and field model.

        Parameters
        ----------
        glyph_size : float, optional
            Length of the glyphs representing the electric field vectors.
            The glyph size is fixed and does not scale with field magnitude.
        """

        self.glyph_size = glyph_size
        self.field = DielectricField()
        self.plotter = pv.Plotter()

    def show(self):
        """
        Render the electric field visualization.

        The method computes the electric field, creates a point cloud,
        attaches vector and magnitude data, and renders oriented glyphs
        colored by field magnitude.

        Returns
        -------
        None
        """

        points, vectors_unit, mag = self.field.calculate_field()

        regions = self.field.regions()

        cloud = pv.PolyData(points)
        cloud["vectors"] = vectors_unit
        cloud["region"] = regions

        mag_gas = np.where(regions == Region.GAS, mag, np.nan)
        mag_diel = np.where(regions == Region.DIELECTRIC, mag, np.nan)

        cloud["mag_gas"] = mag_gas
        cloud["mag_diel"] = mag_diel

        glyphs = cloud.glyph(orient="vectors", scale=False, factor=self.glyph_size)

        glyphs_gas = glyphs.threshold(Region.GAS, scalars="region")
        glyphs_diel = glyphs.threshold(Region.DIELECTRIC, scalars="region")

        self.plotter.add_mesh(
            glyphs_gas,
            scalars="mag_gas",
            cmap="viridis",
            scalar_bar_args={"title": "|E| GAS [V/m]"},
        )

        self.plotter.add_mesh(
            glyphs_diel,
            scalars="mag_diel",
            cmap="plasma",
            scalar_bar_args={"title": "|E| DIELECTRIC [V/m]"},
        )

        error = self.field.mean_radial_error() * 100

        self.plotter.add_text(
            f"Mean radial error:\n{error:.3f} %",
            position=(0.02, 0.9),
            viewport=True,
            font_size=12,
            color="black",
        )

        # Outermost extent of the computed geometry or field
        axis_length = float(np.max(np.linalg.norm(points, axis=1)))

        x_axis = pv.Line((-axis_length, 0, 0), (axis_length, 0, 0))
        y_axis = pv.Line((0, -axis_length, 0), (0, axis_length, 0))
        z_axis = pv.Line((0, 0, -axis_length), (0, 0, axis_length))

        self.plotter.add_mesh(
            x_axis, color="red", line_width=self.LINE_WIDTH, name="x_axis"
        )
        self.plotter.add_mesh(
            y_axis, color="green", line_width=self.LINE_WIDTH, name="y_axis"
        )
        self.plotter.add_mesh(
            z_axis, color="blue", line_width=self.LINE_WIDTH, name="z_axis"
        )

        pv.Plotter.add_axes(self.plotter)
        self.plotter.show()
