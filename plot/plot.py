"""
Author: oQuintino
Description:
    Visualization utilities for rendering the electric field of the coaxial
    dielectric cylinder using PyVista.
"""

import numpy as np
import pyvista as pv
from field import DielectricField, Region


class PlotBuilder:
    """
    Builder class for PyVista-based visualization of an electric field.

    This class encapsulates the logic required to construct a 3D plot
    of the electric field in a coaxial dielectric geometry, including
    vector glyphs, region-specific coloring, coordinate axes, and
    diagnostic text overlays.
    """

    LINE_WIDTH = 1
    """Line width used for rendering coordinate axes."""

    def __init__(self, glyph_size=1e-3):
        """
        Initialize the plot builder.

        Parameters
        ----------
        glyph_size : float, optional
            Fixed length of the glyphs representing electric field vectors.
            The glyph size does not scale with field magnitude.
        """

        self.glyph_size = glyph_size
        self.field = DielectricField()
        self.plotter = pv.Plotter()

    def add_glyphs(self):
        """
        Add electric field glyphs to the plot.

        This method computes the electric field, constructs a point cloud,
        attaches vector, magnitude, and region data, and renders oriented
        glyphs with fixed size.

        Glyphs are colored by field magnitude using distinct colormaps
        for each physical region (gas and dielectric).
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

    def add_axes(self):
        """
        Add Cartesian reference axes to the plot.

        The axes are centered at the origin and extend symmetrically
        based on the current bounds of the plotted geometry or data.
        Colors follow the convention:
        x-axis (red), y-axis (green), z-axis (blue).
        """

        axis_length = np.max(np.abs(self.plotter.bounds))

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

    def add_error_text(self):
        """
        Add a diagnostic text overlay with the mean radial error.

        The error is displayed as a percentage and positioned
        in viewport coordinates, independent of camera movement.
        """

        error = self.field.mean_radial_error() * 100

        self.plotter.add_text(
            f"Mean radial error:\n{error:.3f} %",
            position=(0.02, 0.9),
            viewport=True,
            font_size=12,
            color="black",
        )

    def show(self):
        """
        Render the visualization and start the interactive PyVista window.
        """

        self.plotter.show()
