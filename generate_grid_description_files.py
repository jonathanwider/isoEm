import numpy as np
import os.path

from icosahedron import Icosahedron, rand_rotation_icosahedron, rand_rotation_matrix, plot_voronoi, plot_voronoi_charts


# slightly need to adapt the function from Icosahedron to work with this data format.

def cartesian_to_spherical(data):
    """
    convert cartesian coordinates to spherical coordinates
    Use answer to:
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    # takes list xyz (single coord)
    x = data[..., 0]
    y = data[..., 1]
    z = data[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    # format in HadCM3: lat:(-90,90), lon(0,360)
    theta = 90 - np.arccos(z / r) * 180 / np.pi  # to degrees
    phi = 180 + np.arctan2(y, x) * 180 / np.pi
    return np.array([theta, phi]).transpose((1, 0))  # careful, this will only work if the shape is correct


def generate_grid_descriptions_files(resolution, directory_grids="Grids/", write_bounds=True):
    """
    Generate the grid description files that are required by cdo to do the interpolation.
    @param resolution: Resolution level of the icosahedron
    @param directory_grids: Directory to store the grid description files in.
    @param write_bounds: Determines whether or not we want to write bounding values for the boxes surrounding
                         the centers. Not required for NN-interpolation, but for other remapping schemes.
    """

    name_6nb = "grid_description_r_{}_nbs_6_ico.txt".format(resolution)
    name_5nb = "grid_description_r_{}_nbs_5_ico.txt".format(resolution)

    ico = Icosahedron(r=resolution)

    regions, vertices = ico.get_voronoi_regions_vertices()

    # If we want to generate rotated data, we need to change the next line.
    points = ico.get_charts_cut().reshape(-1, 3)
    regions_six_nb = []
    points_six_nb = []
    indices_six_nb = []
    regions_five_nb = []
    points_five_nb = []
    indices_five_nb = []

    for i in range(len(regions)):
        if len(regions[i]) > 5:
            points_six_nb.append(points[i])
            regions_six_nb.append(regions[i])
            indices_six_nb.append(i)
        else:
            points_five_nb.append(points[i])
            regions_five_nb.append(regions[i])
            indices_five_nb.append(i)

    # create numpy arrays

    regions_six_nb = np.array(regions_six_nb)
    points_six_nb = np.array(points_six_nb)

    regions_five_nb = np.array(regions_five_nb)
    points_five_nb = np.array(points_five_nb)

    # convert to spherical coordinates

    spherical_points_six_nb = cartesian_to_spherical(points_six_nb)
    spherical_points_five_nb = cartesian_to_spherical(points_five_nb)

    spherical_vertices = cartesian_to_spherical(vertices)

    # get the vertices surrounding each point. Should be in ccw rotation by default,
    # because we used sort_vertices_of_regions(). We need ccw rotation because this is required by cdo.

    neighbors_six_nb = np.zeros((spherical_points_six_nb.shape[0], 6, spherical_points_six_nb.shape[1]))
    neighbors_five_nb = np.zeros((spherical_points_five_nb.shape[0], 5, spherical_points_five_nb.shape[1]))

    for i in range(len(spherical_points_six_nb)):
        neighbors_six_nb[i, :, :] = spherical_vertices[regions_six_nb[i]]

    for i in range(len(spherical_points_five_nb)):
        neighbors_five_nb[i, :, :] = spherical_vertices[regions_five_nb[i]]

    # reshape into the format that is required by cdo.

    y_data_six_nb, x_data_six_nb = spherical_points_six_nb.transpose(1, 0)
    y_bounds_six_nb, x_bounds_six_nb = neighbors_six_nb.transpose(2, 0, 1)

    y_data_five_nb, x_data_five_nb = spherical_points_five_nb.transpose(1, 0)
    y_bounds_five_nb, x_bounds_five_nb = neighbors_five_nb.transpose(2, 0, 1)

    with open(os.path.join(directory_grids, name_6nb), 'w') as f:
        f.write('gridtype  = unstructured \n')
        f.write('gridsize  = {}\n'.format(len(spherical_points_six_nb)))
        f.write('nvertex   = 6 \n')

        f.write('xvals     = ')
        np.savetxt(f, x_data_six_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='', comments='# ')

        if write_bounds is True:
            f.write('xbounds   = ')
            np.savetxt(f, x_bounds_six_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='',
                       comments='# ')

        f.write('yvals     = ')
        np.savetxt(f, y_data_six_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='', comments='# ')

        if write_bounds is True:
            f.write('ybounds   = ')
            np.savetxt(f, y_bounds_six_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='',
                       comments='# ')

    if name_5nb is not None:
        with open(os.path.join(directory_grids, name_5nb), 'w') as f:
            f.write('gridtype  = unstructured \n')
            f.write('gridsize  = {}\n'.format(len(spherical_points_five_nb)))
            f.write('nvertex   = 5 \n')

            f.write('xvals     = ')
            np.savetxt(f, x_data_five_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='', comments='# ')

            if write_bounds is True:
                f.write('xbounds   = ')
                np.savetxt(f, x_bounds_five_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='',
                           comments='# ')

            f.write('yvals     = ')
            np.savetxt(f, y_data_five_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='', comments='# ')

            if write_bounds is True:
                f.write('ybounds   = ')
                np.savetxt(f, y_bounds_five_nb, fmt='%.5f', delimiter='  ', newline='\n', header='', footer='',
                           comments='# ')