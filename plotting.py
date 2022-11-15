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

# for plotting maps of R^2 score
r2_style = dict(map_style)
r2_style["CMAP"] = matplotlib.colors.ListedColormap(["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
                                                     "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"])
r2_style["BOUNDS"] = np.concatenate((np.array([-1.0, -0.8, -0.6, -0.4, -0.2]), np.linspace(0, 1, 6)))
r2_style["NORM"] = matplotlib.colors.BoundaryNorm(r2_style["BOUNDS"], r2_style["CMAP"].N)
r2_style["CBAR_LABEL"] = r"$R^2$ score"
r2_style["CBAR_EXTEND"] = "min"

# for plotting maps of isotopic mean state
mean_style = dict(map_style)
mean_style["CMAP"] = matplotlib.colors.ListedColormap(["#003c30", "#01665e", "#35978f",
                                                       "#80cdc1", "#c7eae5", "#f6e8c3"])
mean_style["BOUNDS"] = np.linspace(-30, 6, len(mean_style["CMAP"].colors)+1)
mean_style["NORM"] = matplotlib.colors.BoundaryNorm(mean_style["BOUNDS"], len(mean_style["CMAP"].colors))
mean_style["CBAR_LABEL"] = r"$\delta{}^{18}O$ [‰]"
mean_style["CBAR_EXTEND"] = "both"

# for plotting maps of isotopic std deviation
std_style = dict(map_style)
std_style["CMAP"] = matplotlib.colors.ListedColormap(["#fef0d9", "#fdd49e", "#fdbb84",
                                                      "#fc8d59", "#ef6548", "#d7301f", "#990000"])
std_style["BOUNDS"] = np.linspace(0, 7, len(std_style["CMAP"].colors)+1)
std_style["NORM"] = matplotlib.colors.BoundaryNorm(std_style["BOUNDS"], len(std_style["CMAP"].colors))
std_style["CBAR_LABEL"] = r"$\delta{}^{18}O$ [‰]"
std_style["CBAR_EXTEND"] = "max"

# for plotting maps of correlations
corr_style = dict(map_style)
corr_style["CMAPS"] = {"tsurf": matplotlib.colors.ListedColormap(["#fff", "#A6DCA6", "#70C170", "#3E9E3E"]),
                       "prec": matplotlib.colors.ListedColormap(["#fff", "#FFEECA", "#F0D18F", "#BA9545"]),
                       "slp": matplotlib.colors.ListedColormap(["#fff", "#B59DC5", "#815C99", "#562B71"])}
corr_style["NORM"] = matplotlib.colors.BoundaryNorm(np.linspace(0, 1, 5), 4)
corr_style["CBAR_LABELS"] = {"tsurf": "Temperature",
                             "prec": "Precipitation amount",
                             "slp": "Sea-level pressure"}
corr_style["CBAR_EXTEND"] = "neither"
corr_style["FIGSIZE"] = (15, 9)


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
            extend=style["CBAR_EXTEND"],
            ax=ax)

    cbar.set_label(style["CBAR_LABEL"], fontsize=style["CBAR_FONTSIZE"])

    cbar.ax.tick_params(labelsize=style["CBAR_FONTSIZE"])

    ax.coastlines()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])


def plot_masked_data(ax, data, description, style, title=""):
    """
    Plot data on a 2d grid in a given style.
    @param ax: Axes to plot on
    @param data: Dict of masked fields to be plotted into the same plot.
    @param description: A description of the used dataset. Used to extract latitudes and longitudes.
    @param style: A plotting style (sizes, fonts, etc.)
    @param title: Title of the plot
    @return:
    """

    lat = np.array(description["LATITUDES"])
    lon = np.array(description["LONGITUDES"])

    ax.set_global()

    # remove white line
    fields = []
    lons_plot = []
    for i, key in enumerate(list(data.keys())):
        field, lon_plot = add_cyclic_point(data[key], coord=lon)
        fields.append(field)
        lons_plot.append(lon_plot)

    lo, la = np.meshgrid(lons_plot[0], lat)
    cbars = []
    for i, key in enumerate(list(data.keys())):

        ax.pcolormesh(lo, la, fields[i], transform=ccrs.PlateCarree(), cmap=style["CMAPS"][key], norm=style["NORM"])

    ax.coastlines()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])


def plot_ico_map(ax, data, description, style, title=""):
    """
    Plot icosahedral data in a given style.
    @param ax: Axes to plot on
    @param data: Data to be plotted. Shape must be (n_polygons,)
    @param description: A description of the used dataset. Used to extract latitudes and longitudes.
    @param style: A plotting style (sizes, fonts, etc.)
    @param title: Title of the plot
    @return:
    """
    from icosahedron import Icosahedron
    from util import cartesian_to_spherical

    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    ico = Icosahedron(r=description["RESOLUTION"])
    regions, vertices = ico.get_voronoi_regions_vertices()

    spherical_vertices = cartesian_to_spherical(vertices)
    spherical_vertices_plot = np.zeros_like(spherical_vertices)
    spherical_vertices_plot[:, 0] = spherical_vertices[:, 1]  # longitude
    spherical_vertices_plot[:, 0][spherical_vertices_plot[:, 0] == 360] = 0  # longitude
    spherical_vertices_plot[:, 1] = spherical_vertices[:, 0]  # latitude

    ax.set_global()

    patches = []

    for i in range(len(regions)):
        tmp = spherical_vertices_plot[regions[i]]
        # Polygons that lie close to the 0°-360° continuity get connected wrongly by cartopy. We fix this for now
        # Solution is not perfect.
        if np.amax(tmp[:, 0]) - np.amin(tmp[:, 0]) > 180:
            tmp[tmp > 180] = tmp[tmp > 180] - 360
        polygon = mpatches.Polygon(tmp,
                                   transform=ccrs.PlateCarree())
        polygon.set_color(style["CMAP"](style["NORM"](data[i])))
        ax.add_patch(polygon)

    cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(cmap=style["CMAP"], norm=style["NORM"]),
            spacing='proportional',
            orientation='horizontal',
            extend=style["CBAR_EXTEND"],
            ax=ax)

    cbar.set_label(style["CBAR_LABEL"], fontsize=style["CBAR_FONTSIZE"])

    cbar.ax.tick_params(labelsize=style["CBAR_FONTSIZE"])

    ax.coastlines()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])