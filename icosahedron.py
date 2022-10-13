import numpy as np
from scipy.spatial import SphericalVoronoi


class Icosahedron:
    def __init__(self, c=np.array([0, 0, 0]), rad=1, r=0, perm=np.arange(12)):
        """
        Initialize a given icosahedron (i.e. the collection of its charts). The Icosahedron is initially oriented such
        that one corner is north aligned and that all charts have the north pole in the top left corner (entry 0,0)
        :param c: Center of the icosahedron
        :param rad: Radius of the icosahedron (distance from center to corner).
        :param r: Refinement level, starting at r=0 for standard icosahedron
        """

        assert type(r) == int
        assert 0 <= r < 10
        self.center = np.array([0, 0, 0])
        self.radius = 1
        self.r_level = 0
        # create initial charts - for construction use https://mathworld.wolfram.com/RegularIcosahedron.html
        self.charts = np.zeros((5, 2, 3, 3))
        self.corners = np.zeros_like(self.charts)

        self.perm = perm

        # helping vars taken from link above:
        r_circ = 2 / 5 * np.sqrt(5)
        h_circ = 1 / 5 * np.sqrt(5)  # height (+ and -) of the two circles

        vertices = np.array([[0, 0, 1],
                             [r_circ * np.cos(1 / 5 * (2 * np.pi)), r_circ * np.sin(1 / 5 * (2 * np.pi)), h_circ],
                             [r_circ * np.cos(2 / 5 * (2 * np.pi)), r_circ * np.sin(2 / 5 * (2 * np.pi)), h_circ],
                             [r_circ * np.cos(3 / 5 * (2 * np.pi)), r_circ * np.sin(3 / 5 * (2 * np.pi)), h_circ],
                             [r_circ * np.cos(4 / 5 * (2 * np.pi)), r_circ * np.sin(4 / 5 * (2 * np.pi)), h_circ],
                             [r_circ * np.cos(0 / 5 * (2 * np.pi)), r_circ * np.sin(0 / 5 * (2 * np.pi)), h_circ],
                             [0, 0, -1],
                             [r_circ * np.cos((1 / 10 + 3 / 5) * (2 * np.pi)),
                              r_circ * np.sin((1 / 10 + 3 / 5) * (2 * np.pi)), -h_circ],
                             [r_circ * np.cos((1 / 10 + 4 / 5) * (2 * np.pi)),
                              r_circ * np.sin((1 / 10 + 4 / 5) * (2 * np.pi)), -h_circ],
                             [r_circ * np.cos((1 / 10 + 0 / 5) * (2 * np.pi)),
                              r_circ * np.sin((1 / 10 + 0 / 5) * (2 * np.pi)), -h_circ],
                             [r_circ * np.cos((1 / 10 + 1 / 5) * (2 * np.pi)),
                              r_circ * np.sin((1 / 10 + 1 / 5) * (2 * np.pi)), -h_circ],
                             [r_circ * np.cos((1 / 10 + 2 / 5) * (2 * np.pi)),
                              r_circ * np.sin((1 / 10 + 2 / 5) * (2 * np.pi)), -h_circ]])

        # chart 1
        self.charts[0, 0, 0, :] = vertices[self.perm[0]]
        self.charts[0, 0, 1, :] = vertices[self.perm[1]]
        self.charts[0, 0, 2, :] = vertices[self.perm[10]]
        self.charts[0, 1, 0, :] = vertices[self.perm[5]]
        self.charts[0, 1, 1, :] = vertices[self.perm[9]]
        self.charts[0, 1, 2, :] = vertices[self.perm[6]]

        # chart 2
        self.charts[1, 0, 0, :] = vertices[self.perm[0]]
        self.charts[1, 0, 1, :] = vertices[self.perm[5]]
        self.charts[1, 0, 2, :] = vertices[self.perm[9]]
        self.charts[1, 1, 0, :] = vertices[self.perm[4]]
        self.charts[1, 1, 1, :] = vertices[self.perm[8]]
        self.charts[1, 1, 2, :] = vertices[self.perm[6]]

        # chart 3
        self.charts[2, 0, 0, :] = vertices[self.perm[0]]
        self.charts[2, 0, 1, :] = vertices[self.perm[4]]
        self.charts[2, 0, 2, :] = vertices[self.perm[8]]
        self.charts[2, 1, 0, :] = vertices[self.perm[3]]
        self.charts[2, 1, 1, :] = vertices[self.perm[7]]
        self.charts[2, 1, 2, :] = vertices[self.perm[6]]

        # chart 4
        self.charts[3, 0, 0, :] = vertices[self.perm[0]]
        self.charts[3, 0, 1, :] = vertices[self.perm[3]]
        self.charts[3, 0, 2, :] = vertices[self.perm[7]]
        self.charts[3, 1, 0, :] = vertices[self.perm[2]]
        self.charts[3, 1, 1, :] = vertices[self.perm[11]]
        self.charts[3, 1, 2, :] = vertices[self.perm[6]]

        # chart 5
        self.charts[4, 0, 0, :] = vertices[self.perm[0]]
        self.charts[4, 0, 1, :] = vertices[self.perm[2]]
        self.charts[4, 0, 2, :] = vertices[self.perm[11]]
        self.charts[4, 1, 0, :] = vertices[self.perm[1]]
        self.charts[4, 1, 1, :] = vertices[self.perm[10]]
        self.charts[4, 1, 2, :] = vertices[self.perm[6]]
        self.rescale(rad)
        self.recenter(c)

        self.corners[...] = self.charts[...]

        self.poles = np.array([vertices[self.perm[0]], vertices[self.perm[6]]])

        for i in range(r):
            self.charts = self.refine_charts()

    def calculate_midpoints(self, points_1, points_2):
        """
        Calculate midpoints of edges, where the edges are defined in one point by points_1,
        and the other point by points_2.

        Specify center and radius since there might have been a rotation/scaling/translation in the meantime.
        """
        res = 1 / 2 * (points_2 + points_1)
        res = self.radius * (res - self.center) / np.linalg.norm(res - self.center, axis=-1)[
            ..., np.newaxis] + self.center
        return res

    def refine_charts(self):
        # make sure that the chart has the right shape (first two axis: position in grid, third axis: x,y,z coords)
        assert len(self.charts.shape) == 4
        # make sure that we use euclidean parametrization, eg. 3d coordinates
        assert self.charts.shape[-1] == 3

        # create a new chart of the correct shape
        new_charts = np.zeros((5, 2 * self.charts.shape[1] - 1, 2 * self.charts.shape[2] - 1, 3))
        # copy already created points
        new_charts[:, ::2, ::2, :] = self.charts
        # consider "horizontal" axes of the old chart
        new_charts[:, ::2, 1::2, :] = self.calculate_midpoints(self.charts[:, :, 1:, :], self.charts[:, :, :-1, :])
        # consider "vertical" axes of the old chart
        new_charts[:, 1::2, ::2, :] = self.calculate_midpoints(self.charts[:, 1:, :, :], self.charts[:, :-1, :, :])
        # consider "diagonal" axes of the old chart:
        new_charts[:, 1::2, 1::2, :] = self.calculate_midpoints(self.charts[:, :-1, 1:, :], self.charts[:, 1:, :-1, :])
        self.r_level += 1
        return new_charts

    def n_points(self):
        """Geth number of points for a given refinement level"""
        return 5 * 2 ** (2 * self.r_level + 1) + 2

    def n_faces(self):
        return 20 * 4 ** self.r_level

    def n_edges(self):
        return self.n_points + self.n_edges - 2

    def spherical_to_cartesian(self, charts_spherical):
        """
        convert spherical coordinates to euclidean coordinates.
        Use answer to:
        https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
        """
        # takes list rthetaphi (single coord)
        r = charts_spherical[..., 0]
        theta = - (charts_spherical[..., 1] - 90) * (np.pi / 180)
        phi = (charts_spherical[..., 2] - 180) * (np.pi / 180)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z]).transpose((1, 2, 3, 0))

    def cartesian_to_spherical(self):
        """
        convert cartesian coordinates to spherical coordinates
        Use answer to:
        https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
        """
        # takes list xyz (single coord)
        x = self.charts[..., 0]
        y = self.charts[..., 1]
        z = self.charts[..., 2]
        r = np.sqrt(x * x + y * y + z * z)
        # format in HadCM3: lat:(-90,90), lon(0,360)
        theta = 90 - np.arccos(z / r) * 180 / np.pi  # to degrees
        phi = 180 + np.arctan2(y, x) * 180 / np.pi
        return np.array([r, theta, phi]).transpose((1, 2, 3, 0))  # careful, this will only work if the shape is correct

    def get_rotated_charts_cut(self, rot):
        """
        Rotate the icosahedron (all points) with a given rotation matrix.
        """
        return np.einsum('ij,chwj->chwi', rot, self.charts)[..., 1:, :-1, :]

    def recenter(self, new_center):
        """
        Translate the origin of the icosahedron
        :param new_center: New center of the icosahedron (should be numpy array of shape (3,))
        """
        assert len(new_center) == 3
        assert len(new_center.shape) == 1

        self.charts = self.charts - self.center + new_center
        self.center = new_center

    def rescale(self, new_radius):
        """
        Rescale the radius of the icosahedron
        :param new_radius: New radius (should be >0)
        """
        assert new_radius > 0
        self.charts = (self.charts - self.center) * (new_radius / self.radius) + self.center
        self.radius = new_radius

    def get_charts_cut(self):
        """In the network we don't use the whole charts defined here but remove one row and one column each.
        Doing this gets rid of double-points and the points ad N and S."""
        return self.charts[..., 1:, :-1, :]

    def get_voronoi_regions_vertices(self, rot=None):
        """
        Get the voronoi regions around each point in the charts. Useful for plotting
        """
        poles = self.radius * self.poles + self.center

        if rot is None:
            points = self.get_charts_cut().reshape(-1, 3)
        else:
            points = self.get_rotated_charts_cut(rot).reshape(-1, 3)
            poles = np.einsum('ij,kj->ki', rot, poles)

        points = np.concatenate((poles, points), axis=0)

        sv = SphericalVoronoi(points, self.radius, self.center)
        sv.sort_vertices_of_regions()
        # here the vertices are sorted in EITHER cw or ccw rotation. We NEED counterclockwise rotation

        north = poles[0, ...]
        counts = np.zeros(len(points))

        for i, point in enumerate(points[2:]):
            neighbors = sv.vertices[sv.regions[i + 2]]  # +2 because we need to drop the poles

            # calculate normal vectors
            x = np.cross(north, point)
            y = np.cross(point, x)

            # calculate coordinates in this basis
            x_n = np.dot(neighbors, x)
            y_n = np.dot(neighbors, y)

            points_tangent = np.array([x_n, y_n])
            points_tangent = points_tangent / np.linalg.norm(points_tangent, axis=0)[np.newaxis, ...]

            angle = np.arctan2(points_tangent[0, ...], points_tangent[1, ...]) * 360 / 2 / np.pi
            rolled_angle = np.roll(angle, axis=0, shift=1)
            diff = angle - rolled_angle

            greater = diff > 0
            counts[i + 2] = np.sum(greater, axis=-1)
        regions_sorted = []
        for i in range(len(sv.regions)):
            if counts[i] == 1:
                regions_sorted.append(sv.regions[i])
            elif counts[i] > 1:
                regions_sorted.append(np.flip(sv.regions[i], ))
            else:
                regions_sorted.append(sv.regions[i])

        return regions_sorted[2:], sv.vertices


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    # From UGSCNN-Code
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rand_rotation_icosahedron():
    """
    Rotate the icosahedron randomly by a rotational symmetry. This is implemented by applying a permutation
    to the stored corner indices
    """

    def antipode(index):
        return (index + 6) % 12

    # the neighborhood of each vertex in clockwise rotation:
    neighbor_indices = np.array([[1, 2, 3, 4, 5],
                                 [0, 5, 9, 10, 2],
                                 [0, 1, 10, 11, 3],
                                 [0, 2, 11, 7, 4],
                                 [0, 3, 7, 8, 5],
                                 [0, 4, 8, 9, 1],
                                 [11, 10, 9, 8, 7],
                                 [6, 8, 4, 3, 11],
                                 [6, 9, 5, 4, 7],
                                 [6, 10, 1, 5, 8],
                                 [6, 11, 2, 1, 9],
                                 [6, 7, 3, 2, 10]])

    # decide ranomly which point becomes new north pole and in what rotation it neighborhood should be
    new_NP = np.random.randint(12)
    rot_state = np.random.randint(5)

    NP_neighbors = np.roll(neighbor_indices[new_NP, ...], shift=rot_state)

    new_perm = np.concatenate(([new_NP], NP_neighbors, [antipode(new_NP)], antipode(NP_neighbors)))
    return new_perm


def all_rotations_icosahedron():
    """
    Get an array containing all possible rotations of the icosahedron.
    """

    def antipode(index):
        return (index + 6) % 12

    # the neighborhood of each vertex in clockwise rotation:
    neighbor_indices = np.array([[1, 2, 3, 4, 5],
                                 [0, 5, 9, 10, 2],
                                 [0, 1, 10, 11, 3],
                                 [0, 2, 11, 7, 4],
                                 [0, 3, 7, 8, 5],
                                 [0, 4, 8, 9, 1],
                                 [11, 10, 9, 8, 7],
                                 [6, 8, 4, 3, 11],
                                 [6, 9, 5, 4, 7],
                                 [6, 10, 1, 5, 8],
                                 [6, 11, 2, 1, 9],
                                 [6, 7, 3, 2, 10]])

    # decide ranomly which point becomes new north pole and in what rotation it neighborhood should be
    all_perms = np.zeros((12*5, 12), dtype='int')
    for new_NP in range(12):
        for rot_state in range(5):
            NP_neighbors = np.roll(neighbor_indices[new_NP, ...], shift=rot_state)
            all_perms[5*new_NP + rot_state] = np.concatenate(([new_NP], NP_neighbors, [antipode(new_NP)], antipode(NP_neighbors)))
    return all_perms

def plot_voronoi(data, regions, vertices, cmap=None, norm=None, elev=0, azim=0, figsize=(10, 10)):
    """ Plot flattened data on the sphere using voronoi regions. Display data colorcoded"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if norm is None:
        dmin = np.amin(data)
        dmax = np.amax(data)
    for i in range(len(regions)):
        polygon = Poly3DCollection([vertices[regions[i]]], alpha=1.0)
        if norm is None:
            polygon.set_color(cmap(np.array((data[i]-dmin)/(dmax-dmin))))
        else:
            polygon.set_color(cmap(norm(data[i])))
        ax.add_collection3d(polygon)

    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    ax.view_init(elev, 180 + azim)


def plot_voronoi_charts(data, regions, vertices, label="", elev=0, azim=0, figsize=(10, 10)):
    """
    Plot flattened data on the sphere using voronoi regions. Display data colorcoded. Split different maps.
    Assume data has shape (n_charts, height, width)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    assert len(data.shape) == 3
    assert data.shape[0] == 5

    # plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    dmin = np.amin(data)
    dmax = np.amax(data)
    cmaps = ["Reds", "Greens", "Blues", "Greys", "Purples"]

    n_points_per_chart = data.shape[1] * data.shape[2]
    for j in range(data.shape[0]):
        cmap = plt.get_cmap(cmaps[j])
        for i in range(n_points_per_chart):
            reg = regions[j * n_points_per_chart + i]
            polygon = Poly3DCollection([vertices[reg]], alpha=1.0)
            if dmin != dmax:
                c = (data.reshape(5, -1)[j, i] - dmin) / (dmax - dmin)
            else:
                c = data.reshape(5, -1)[j, i]
            polygon.set_color(cmap(c))
            ax.add_collection3d(polygon)
    if label != "":
        ax.set_title(label)
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    ax.view_init(elev, 180 + azim)
