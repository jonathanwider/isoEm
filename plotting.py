import cartopy
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point


class FixPointNormalize(matplotlib.colors.Normalize):
    """
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level"
    to a color in the blue/turquise range.
    """

    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val=0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


timeseries_style = {
    "ASPECT_RATIO": 2,
    "COLOR": "#31a354",
    "TITLE_FONTSIZE": 8,
}

map_style = {
    "ASPECT_RATIO": 7/5,
    "PROJECTION": ccrs.Robinson(),
    "TITLE_FONTSIZE": 8,
}

# for plotting maps of R^2 score
r2_style = dict(map_style)

r2_style["CMAP"] = matplotlib.colors.ListedColormap(
    ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"])
r2_style["BOUNDS"] = np.concatenate(
    (np.array([-1.0, -0.8, -0.6, -0.4, -0.2]), np.linspace(0, 1, 6)))
r2_style["NORM"] = matplotlib.colors.BoundaryNorm(
    r2_style["BOUNDS"], r2_style["CMAP"].N)
r2_style["CBAR_LABEL"] = r"$R^2$ score"
r2_style["CBAR_EXTEND"] = "min"

# for plotting maps of isotopic mean state
mean_style = dict(map_style)
mean_style["CMAP"] = matplotlib.colors.ListedColormap(["#003c30", "#01665e", "#35978f",
                                                       "#80cdc1", "#c7eae5", "#f6e8c3"])
mean_style["BOUNDS"] = np.linspace(-30, 6, len(mean_style["CMAP"].colors) + 1)
mean_style["NORM"] = matplotlib.colors.BoundaryNorm(
    mean_style["BOUNDS"], len(mean_style["CMAP"].colors))
mean_style["CBAR_LABEL"] = r"$\delta{}^{18}$O [‰]"
mean_style["CBAR_EXTEND"] = "both"

# for plotting maps of isotopic std deviation
std_style = dict(map_style)
std_style["CMAP"] = matplotlib.colors.ListedColormap(["#fef0d9", "#fdd49e", "#fdbb84",
                                                      "#fc8d59", "#ef6548", "#d7301f", "#990000"])
std_style["BOUNDS"] = np.linspace(
    0, 7, len(std_style["CMAP"].colors) + 1)
std_style["NORM"] = matplotlib.colors.BoundaryNorm(
    std_style["BOUNDS"], len(std_style["CMAP"].colors))
std_style["CBAR_LABEL"] = r"$\delta{}^{18}$O [‰]"
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
corr_style["ASPECT_RATIO"] = 9/5


# for plotting rms differences in correlations:
corr_diff_style = dict(map_style)
corr_diff_style["CMAP"] = matplotlib.colors.ListedColormap(["#edf8fb", "#b3cde3", "#8c96c6", "#8856a7", "#810f7c"])
corr_diff_style["NORM"] = matplotlib.colors.Normalize(vmin=0, vmax=1)
corr_diff_style["CBAR_LABEL"] = r"$l^2$ norm intermodel correlation difference"
corr_diff_style["CBAR_EXTEND"] = "max"

# for plotting maps of temperature
tsurf_style = dict(map_style)
tsurf_style["CMAP"] = plt.get_cmap("RdBu_r")
tsurf_style["NORM"] = matplotlib.colors.TwoSlopeNorm(
    vmin=-40, vmax=40, vcenter=0)
tsurf_style["CBAR_LABEL"] = "Temperature [K]"
tsurf_style["CBAR_EXTEND"] = "both"

# for plotting maps of precipitation
prec_style = dict(map_style)
prec_style["CMAP"] = plt.get_cmap("YlGnBu")
prec_style["NORM"] = matplotlib.colors.Normalize(vmin=0, vmax=800)
prec_style["CBAR_LABEL"] = "Precipitation [mm/month]"
prec_style["CBAR_EXTEND"] = "both"

# for plotting MNIST-digits
digit_style = dict(map_style)
digit_style["CMAP"] = plt.get_cmap("viridis")
digit_style["ASPECT_RATIO"] = 1
digit_style["NORM"] = matplotlib.colors.Normalize(vmin=0, vmax=256)

# for plotting MNIST-digits
anom_style = dict(map_style)
anom_style["CMAP"] = plt.get_cmap("BrBG")
anom_style["ASPECT_RATIO"] = 2
anom_style["NORM"] = matplotlib.colors.Normalize(vmin=-3, vmax=3)
anom_style["CBAR_EXTEND"] = "both"
anom_style["CBAR_LABEL"] = r"$\delta{}^{18}$O anomaly [‰]"

diff_coarse_style = dict(map_style)
diff_coarse_style["CMAP"] = plt.get_cmap("PiYG")
diff_coarse_style["NORM"] = matplotlib.colors.Normalize(vmin=-1, vmax=1)
diff_coarse_style["CBAR_LABEL"] = r"$R^2$ score difference"
diff_coarse_style["CBAR_EXTEND"] = "both"

diff_style = dict(map_style)
diff_style["CMAP"] = plt.get_cmap("PiYG")
diff_style["NORM"] = matplotlib.colors.Normalize(vmin=-0.4, vmax=0.4)
diff_style["CBAR_LABEL"] = r"$R^2$ score difference"
diff_style["CBAR_EXTEND"] = "both"

diff_fine_style = dict(map_style)
diff_fine_style["CMAP"] = plt.get_cmap("PiYG")
diff_fine_style["NORM"] = matplotlib.colors.Normalize(vmin=-0.12, vmax=0.12)
diff_fine_style["CBAR_LABEL"] = r"$R^2$ score difference"
diff_fine_style["CBAR_EXTEND"] = "both"

std_anom_style = dict(map_style)
std_anom_style["CMAP"] = plt.get_cmap("PiYG")
std_anom_style["NORM"] = matplotlib.colors.Normalize(vmin=-1, vmax=1)
std_anom_style["CBAR_LABEL"] = r"Standardized data"
std_anom_style["CBAR_EXTEND"] = "both"

std_anom_style_d18O = dict(map_style)
std_anom_style_d18O["CMAP"] = plt.get_cmap("PuOr")
std_anom_style_d18O["NORM"] = matplotlib.colors.Normalize(vmin=-1, vmax=1)
std_anom_style_d18O["CBAR_LABEL"] = r"Standardized data"
std_anom_style_d18O["CBAR_EXTEND"] = "both"

percentage_style = dict(map_style)
percentage_style["CMAP"] = matplotlib.colors.ListedColormap(["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#990000"])

percentage_style["BOUNDS"] = np.array([0, 1, 5, 10, 15, 20, 25, 30])
percentage_style["NORM"] = matplotlib.colors.BoundaryNorm(percentage_style["BOUNDS"], len(percentage_style["CMAP"].colors))
percentage_style["CBAR_LABEL"] = r"Fraction of missiong $\delta{}^{18}$O values [%]"
percentage_style["CBAR_EXTEND"] = "max"


def plot_map(ax, data, description, style, title="", cbar_orientation='horizontal', show_colorbar=True, show_colorbar_label=True, rasterized=True):
    """
    Plot data on a 2d grid in a given style.
    @param ax: Axis to plot on.
    @param data: Data to be plotted. Shape has to allign with latitudes and longitudes given in description
    @param description: A description of the used dataset. Used to extract latitudes and longitudes.
    @param style: A plotting style (sizes, fonts, etc.)
    @param title: Title of the plot
    @param cbar_orientation: Orientation of the colorbar.
    @param show_colorbar: Whether or not to show a colorbar.
    @param show_colorbar_label: Whether or not to show a colorbar label
    @param rasterized: Whether or not to rasterize the colormesh.
    @return:
    """
    lat = np.array(description["LATITUDES"])
    lon = np.array(description["LONGITUDES"])
    assert data.shape == (len(lat), len(lon))

    ax.set_global()
    # remove white line
    field, lon_plot = add_cyclic_point(data, coord=lon)
    lo, la = np.meshgrid(lon_plot, lat)
    layer = ax.pcolormesh(lo, la, field, transform=ccrs.PlateCarree(
    ), cmap=style["CMAP"], norm=style["NORM"], rasterized=rasterized)

    if show_colorbar:
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(
                cmap=style["CMAP"], norm=style["NORM"]),
            spacing='proportional',
            orientation=cbar_orientation,
            extend=style["CBAR_EXTEND"],
            ax=ax)
        if show_colorbar_label:
            cbar.set_label(style["CBAR_LABEL"])

    ax.coastlines()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])


def find_gridbox(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_map_markers(ax, locs, data, description, style, title="", locs_labels=None, rasterized=True):
    """
    Plot data on a 2d grid in a given style.
    @param ax: Axis to plot on.
    @param locs: List of locations to put markers at.
    @param data: Data to be plotted. Shape has to allign with latitudes and longitudes given in description
    @param description: A description of the used dataset. Used to extract latitudes and longitudes.
    @param style: A plotting style (sizes, fonts, etc.)
    @param title: Title of the plot
    @param locs_labels: List of labels for the locations. If not provided, don't display a legend
    @param rasterized: Whether or not to rasterize the colormesh.
    @return:
    """
    lat = np.array(description["LATITUDES"])
    lon = np.array(description["LONGITUDES"])
    assert data.shape == (len(lat), len(lon))

    # find the indices of the grid boxes in which we want to plot markers.
    locs_boxes = np.zeros_like(locs, dtype='int')
    for i, loc in enumerate(locs):
        locs_boxes[i, 0] = int(find_gridbox(lat, loc[0]))
        locs_boxes[i, 1] = int(find_gridbox(lon, loc[1]))

    # define markers that can be used:
    markers = ["*", "P", "o", "D", "p"]

    ax.set_global()
    # remove white line
    field, lon_plot = add_cyclic_point(data, coord=lon)
    lo, la = np.meshgrid(lon_plot, lat)
    layer = ax.pcolormesh(lo, la, field, transform=ccrs.PlateCarree(
    ), cmap=style["CMAP"], norm=style["NORM"], rasterized=rasterized)

    for i in range(len(locs_boxes)):
        if locs_labels == None:
            ax.plot(locs[i, 1], locs[i, 0], marker=markers[i], linestyle="None",
                    color="k", transform=ccrs.Geodetic())
        else:
            ax.plot(locs[i, 1], locs[i, 0], marker=markers[i], linestyle="None",
                    color="k", transform=ccrs.Geodetic(), label=locs_labels[i])

    cbar = plt.colorbar(
        matplotlib.cm.ScalarMappable(
            cmap=style["CMAP"], norm=style["NORM"]),
        spacing='proportional',
        orientation='vertical',
        extend=style["CBAR_EXTEND"],
        ax=ax)

    cbar.set_label(style["CBAR_LABEL"])

    ax.coastlines()
    # if locs_labels != None:
    #     ax.legend()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])


def plot_masked_data(ax, data, description, style, title="", rasterized=True):
    """
    Plot data on a 2d grid in a given style.
    @param ax: Axes to plot on
    @param data: Dict of masked fields to be plotted into the same plot.
    @param description: A description of the used dataset. Used to extract latitudes and longitudes.
    @param style: A plotting style (sizes, fonts, etc.)
    @param title: Title of the plot
    @param rasterized: Whether or not to rasterize the colormesh.
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
        ax.pcolormesh(lo, la, fields[i], transform=ccrs.PlateCarree(
        ), cmap=style["CMAPS"][key], norm=style["NORM"], rasterized=rasterized)

    ax.coastlines()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])


def plot_ico_map(ax, data, description, style, title="", show_colorbar=True, rasterized=True):
    """
    Plot icosahedral data in a given style.
    @param ax: Axes to plot on
    @param data: Data to be plotted. Shape must be (n_polygons,)
    @param description: A description of the used dataset. Used to extract latitudes and longitudes.
    @param style: A plotting style (sizes, fonts, etc.)
    @param title: Title of the plot
    @param show_colorbar: Whether or not to show a colorbar.
    @param rasterized: Whether or not to rasterize the colormesh.
    @return:
    """
    from icosahedron import Icosahedron
    from util import cartesian_to_spherical

    import matplotlib.patches as mpatches

    ico = Icosahedron(r=description["RESOLUTION"])
    regions, vertices = ico.get_voronoi_regions_vertices()

    spherical_vertices = cartesian_to_spherical(vertices)
    spherical_vertices_plot = np.zeros_like(spherical_vertices)
    spherical_vertices_plot[:, 0] = spherical_vertices[:, 1]
    # longitude

    spherical_vertices_plot[:, 0][spherical_vertices_plot[:, 0] == 360] = 0
    spherical_vertices_plot[:, 1] = spherical_vertices[:, 0]

    ax.set_global()

    patches = []

    for i in range(len(regions)):
        tmp = spherical_vertices_plot[regions[i]]
        # Polygons that lie close to the 0°-360° continuity get connected wrongly by cartopy. We fix this for now
        # Solution is not perfect.
        if np.amax(tmp[:, 0]) - np.amin(tmp[:, 0]) > 180:
            tmp[tmp > 180] = tmp[tmp > 180] - 360
        polygon = mpatches.Polygon(tmp,
                                   transform=ccrs.PlateCarree(), rasterized=rasterized)
        polygon.set_color(style["CMAP"](style["NORM"](data[i])))
        ax.add_patch(polygon)

    if show_colorbar:
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(
                cmap=style["CMAP"], norm=style["NORM"]),
            spacing='proportional',
            orientation='horizontal',
            extend=style["CBAR_EXTEND"],
            ax=ax)

        cbar.set_label(style["CBAR_LABEL"])

    ax.coastlines()

    ax.set_title(title, fontsize=style["TITLE_FONTSIZE"])


def get_coastline_xyz(r=1.):
    coords = []
    for g in cartopy.feature.COASTLINE.geometries():
        lon = np.array(g.coords)[:, 0]
        lat = np.array(g.coords)[:, 1]

        theta = - (lat - 90) * (np.pi / 180)
        phi = (lon - 180) * (np.pi / 180)

        # theta = (lat + 90) * np.pi / 180
        # phi = (lon + 180) * 2*np.pi / 360

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        coords.append(np.array([x, y, z]))

    return coords


def plot_map_3d(ax, data, description, style, title="", elev=18, azim=0, show_coastlines=True, show_colorbar=True):
    """
    Plot data of an icosahedral grid in 3d.
    @param ax: Axes to plot on. projection='3d' needs to be set
    @param data: Data to be plotted. Assumed to be of shape (n_pixels_on_icosahedron,)
    @param description: Description of the dataset used
    @param style: Plotting style
    @param title: Title of the plot
    @param elev: Elevation of view position
    @param azim:Azimuth of view position
    @param show_coastlines: Whether or not to display coastline
    @param show_colorbar: Whether to display a colorbar or not
    """

    from icosahedron import Icosahedron
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ico = Icosahedron(r=description["RESOLUTION"])
    regions, vertices = ico.get_voronoi_regions_vertices()

    for i in range(len(regions)):
        polygon = Poly3DCollection([vertices[regions[i]]], alpha=1)
        polygon.set_color(style["CMAP"](style["NORM"](data[i])))
        ax.add_collection3d(polygon)

    if show_coastlines:
        # value of r > 1 so that coastlines don't overlap with hexagons.
        cls = get_coastline_xyz(r=1.016)
        for cl in cls:
            points = np.transpose(cl)
            for i in range(len(points) - 1):
                polygon = Poly3DCollection([points[i:i + 2, :]], alpha=1)
                polygon.set_color("black")
                ax.add_collection3d(polygon)

    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    ax.view_init(elev, 180 + azim)

    if show_colorbar:
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(
                cmap=style["CMAP"], norm=style["NORM"]),
            spacing='proportional',
            orientation='vertical',
            extend=style["CBAR_EXTEND"],
            ax=ax)

        cbar.set_label(style["CBAR_LABEL"])


def plot_timeseries(ax, data_pred, data_gt, loc, loc_label, description, style):
    """
    For a given location plot the testset-timeseries at the closest gridbox.
    Mark missing values.

    Assume that the gt timeseries has shape (time, n_t_vars, lat, lon) and prediction timeseries has shape (n_runs, n_t_vars, time, lat, lon).
    """
    from evaluate import get_r2, get_correlation
    lat = np.array(description["LATITUDES"])
    lon = np.array(description["LONGITUDES"])

    # find the indices of the grid boxes in which we want to plot markers.
    loc_box = np.zeros(2, dtype='int')
    loc_box[0] = int(find_gridbox(lat, loc[0]))
    loc_box[1] = int(find_gridbox(lon, loc[1]))

    r2 = np.zeros(
        (data_pred.shape[0], data_pred.shape[3], data_pred.shape[4]))
    cor = np.zeros(
        (data_pred.shape[0], data_pred.shape[3], data_pred.shape[4]))
    for i in range(len(r2)):
        r2[i] = get_r2(data_pred[i], data_gt)
        cor[i] = get_correlation(data_pred[i], data_gt)

    max_pred = np.amax(data_pred, axis=0)
    min_pred = np.amin(data_pred, axis=0)
    mean_pred = np.mean(data_pred, axis=0)

    ax.plot(mean_pred[:, 0, loc_box[0], loc_box[1]],
            label='mean, emulation', color=style["COLOR"], linewidth=0.8)
    # ax.fill_between(np.arange(len(mean_pred[:, 0, loc_box[0], loc_box[1]])), min_pred[:, 0, loc_box[0], loc_box[1]], max_pred[:, 0, loc_box[0], loc_box[1]],
    #                 label='min-max emulation', color=style["COLOR"], alpha=0.9, linewidth=0.0)
    ax.plot(data_gt[:, 0, loc_box[0], loc_box[1]],
            label='ground truth', color='k', linewidth=0.8)

    metric_mean = np.mean(r2, axis=0)
    metric_std = np.std(r2, axis=0)
    cor_mean = np.mean(cor, axis=0)
    cor_std = np.std(cor, axis=0)

    """
    ax.text(0.55, 0.1, r"Corr: {:0.3f} +/- {:0.3f}, $R^2$score: {:0.3f} +/- {:0.3f}".format(cor_mean[loc_box[0], loc_box[1]], cor_std[loc_box[0], loc_box[1]],
                                                                                                   metric_mean[loc_box[0], loc_box[1]], metric_std[loc_box[0], loc_box[1]]),
    
    """
    ax.text(0.01, 0.01, r"Corr: {:0.2f} +/- {:0.2f}".format(cor_mean[loc_box[0], loc_box[1]], cor_std[loc_box[0], loc_box[1]]),
            horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    ax.set_title("Time series at {}".format(
        loc_label), fontsize=style["TITLE_FONTSIZE"])

    plt.xlabel("timestep in test set")
    plt.ylabel(r"$\delta{}^{18}$O [‰]")


def add_label_to_axes(ax, label, style, fontsize=None):
    if fontsize is None:
        ax.text(.01, .99, label, ha='left', va='top',  
            transform=ax.transAxes)
    else:
        ax.text(.01, .99, label, ha='left', va='top',  
            transform=ax.transAxes, fontsize=fontsize)
