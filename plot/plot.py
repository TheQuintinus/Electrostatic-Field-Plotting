import pyvista as pv

import field


class Plot:
    """
    PyVista plot of the electric field using fixed-size glyphs and magnitude-based coloring.
    """

    def __init__(self, glyph_size=8e-4):
        self.glyph_size = glyph_size
        self.field = field.DielectricField()
        self.plotter = pv.Plotter()

    def show(self):
        points, vectors_unit, mag = self.field.calculate_field()

        cloud = pv.PolyData(points)
        cloud["vectors"] = vectors_unit
        cloud["mag"] = mag

        glyphs = cloud.glyph(orient="vectors", scale=False, factor=self.glyph_size)

        self.plotter.add_mesh(
            glyphs,
            scalars="mag",
            cmap="viridis",
            scalar_bar_args={"title": "|E| [V/m]"},
        )

        self.plotter.add_axes()
        self.plotter.show()
