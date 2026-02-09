import plot


def main():
    plt = plot.PlotBuilder()

    plt.set_isometric_z_right()

    plt.add_error_text()
    plt.add_glyphs()
    plt.add_axes()

    plt.show()


if __name__ == "__main__":
    main()
