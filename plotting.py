import matplotlib.pyplot as plt
import matplotlib

import numpy as np

import cartopy
import cartopy.mpl.geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

map_style = {
    "FIGSIZE": (15, 12),
    "CBAR_FONTSIZE": 12,
    "PROJECTION": ccrs.Robinson(),
    "TITLE_FONTSIZE": 15
}

r2_style = dict(map_style)

r2_style["CMAP"] = matplotlib.colors.ListedColormap(["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
                                                     "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"])
r2_style["BOUNDS"] = np.concatenate((np.array([-1.0, -0.8, -0.6, -0.4, -0.2]), np.linspace(0, 1, 6)))
r2_style["NORM"] = matplotlib.colors.BoundaryNorm(r2_style["BOUNDS"], r2_style["CMAP"].N)
r2_style["CBAR_LABEL"] = r"$R^2$ score"


def plot_map(ax, data, description, style, title=""):
    """
    Plot data on a 2d grid in a given style.
    @param data: Data to be plotted. Shape has to allign with latitudes and longitudes given in description
    @param description: A description of the used dataset. Used to extract latitudes and longitudes.
    @param style: A plotting style (sizes, fonts, etc.)
    @param title: Title of the plot
    @return:
    """
    lat = np.array(description["LATITUDES"])
    lon = np.array(description["LONGITUDES"])
    assert data.shape == (len(lat), len(lon))

    ax.set_global()
    # remove white line
    field, lon_plot = add_cyclic_point(data, coord=lon)
    lo, la = np.meshgrid(lon_plot, lat)
    layer = ax.pcolormesh(lo, la, field, transform=ccrs.PlateCarree(), cmap=style["CMAP"], norm=style["NORM"])

    cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(cmap=style["CMAP"], norm=style["NORM"]),
            spacing='proportional',
            orientation='horizontal',
            ax=ax)

    cbar.set_label(style["CBAR_LABEL"], fontsize=style["CBAR_FONTSIZE"])

    cbar.ax.tick_params(labelsize=style["CBAR_FONTSIZE"])

    ax.coastlines()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])