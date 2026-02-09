"""
Author: oQuintino
Description:
    Visualization utilities for rendering the electric field of the coaxial
    dielectric cylinder using PyVista.
"""

from functools import cached_property

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

    @cached_property
    def cloud(self):
        """
        Cached point cloud containing all data required for visualization.

        This property computes the electric field once and constructs a
        PyVista PolyData object holding:
        - Cartesian coordinates of the grid points
        - Unit electric field vectors
        - Field magnitude
        - Region classification labels
        - Region-specific masked magnitudes
        """

        points, vectors_unit, mag = self.field.calculate_field()
        regions = self.field.regions().astype(np.int8)

        cloud = pv.PolyData(points)

        # Vector data
        cloud["vectors"] = vectors_unit

        # Global magnitude
        cloud["magnitude"] = mag

        # Region labels
        cloud["region"] = regions

        # Region-specific magnitudes (NaN-masked)
        cloud["mag_gas"] = np.where(regions == Region.GAS, mag, np.nan)
        cloud["mag_diel"] = np.where(regions == Region.DIELECTRIC, mag, np.nan)

        return cloud

    def add_glyphs(self):
        """
        Add electric field glyphs to the plot using the cached cloud.

        This method renders oriented glyphs at the cloud points using the
        unit electric field vectors. Glyph size is fixed and independent
        of field magnitude.

        Field magnitude is encoded by color, with separate colormaps
        applied to each physical region (gas and dielectric) based on
        region-specific scalar fields stored in the cloud.
        """

        glyphs = self.cloud.glyph(orient="vectors", scale=False, factor=self.glyph_size)

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

        The axis extents are determined from the spatial extent of the
        cached cloud, ensuring that the axes are well-scaled and independent
        of the order in which plot elements are added.

        Axes are centered at the origin and colored according to the
        standard convention:
        x-axis (red), y-axis (green), z-axis (blue).
        """

        axis_length = np.max(np.abs(self.cloud.points))

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

    def set_isometric_z_right(self):
        """
        Set an isometric view where the physical Z axis points to the right
        on the screen.
        """

        pv.Plotter.enable_parallel_projection(self.plotter)

        cam = self.plotter.camera

        # Camera looks diagonally at the origin
        cam.focal_point = (0, 0, 0)
        cam.position = (1, -1, 1)

        # Screen-up is +Y, so screen-right becomes +Z
        cam.up = (0, 1, 0)

        cam.zoom(1.2)

    def show(self):
        """
        Render the visualization and start the interactive PyVista window.
        """

        self.plotter.show()
