"""Module to generate the spherical mnist data set - taken from S2CNN paper and modified"""

import gzip
import pickle
import numpy as np
import argparse
from icosahedron import Icosahedron, rand_rotation_matrix, all_rotations_icosahedron

NORTHPOLE_EPSILON = 1e-3


def project_sphere_on_xy_plane(grid, projection_origin):
    """ returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere """

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1  # assumes a radius of 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry


def sample_within_bounds(signal, x, y, bounds):
    """ """
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    """ """

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.astype(int)
    iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def project_2d_on_sphere(signal, grid, projection_origin=None):
    """ """
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(signal, rx, ry)

    # ensure that only south hemisphere gets projected
    sample *= (grid[2] <= 1).astype(np.float64)

    # rescale signal to [0,1]
    sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
    sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

    sample = (sample - sample_min) / (sample_max - sample_min)
    sample *= 255
    sample = sample.astype(np.uint8)

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--non_default_setup", dest="default_setup", action="store_false",
                        help="If this flag isn't set, just use a default setup and produce all required MNIST datasets as in paper.")
    parser.add_argument("--refinement",
                        help="the refinement level of the icosahedral grid",
                        type=int,
                        default=4,
                        required=False)
    parser.add_argument("--noise",
                        help="the rotational noise applied on the sphere",
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument("--chunk_size",
                        help="size of image chunk with same rotation",
                        type=int,
                        default=500,
                        required=False)
    parser.add_argument("--mnist_data_folder",
                        help="folder for saving the mnist data",
                        type=str,
                        default="MNIST_data",
                        required=False)
    parser.add_argument("--output_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="ico_mnist.gz",
                        required=False)
    parser.add_argument("--rot_type_train",
                        help="Type of rotations to be applied to the train set",
                        default='no_rot',
                        choices=['no_rot', 'full_rot', 'ico_rot'])
    parser.add_argument("--rot_type_test",
                        help="Type of rotations to be applied to the test set",
                        default='no_rot',
                        choices=['no_rot', 'full_rot', 'ico_rot'])

    args = parser.parse_args()

    print("getting mnist data")
    from torchvision import datasets
    """
    datasets.MNIST.resources = [
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
         'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
         'd53e105ee54ea40749a09fcbcd1e9432'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
         '9fb629c4189551a2d022fa330f9573f3'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
         'ec29112dd5afa0611ce80d1b7f02629c')
    ]
    """
    trainset = datasets.MNIST(
        root=args.mnist_data_folder, train=True, download=True)
    testset = datasets.MNIST(root=args.mnist_data_folder,
                             train=False, download=True)
    mnist_train = {'images': trainset.data.numpy(
    ), 'labels': trainset.targets.numpy()}
    mnist_test = {'images': testset.data.numpy(
    ), 'labels': testset.targets.numpy()}

    ico = Icosahedron(r=args.refinement, rad=1, c=np.array([0, 0, 0]))
    grid = ico.get_charts_cut()
    grid = grid.reshape(
        grid.shape[0] * grid.shape[1], grid.shape[2], grid.shape[3])

    if args.default_setup:
        print("Use default setup for producing spherical MNIST dataset")
        configs = [["no_rot", "no_rot"], ["no_rot", "ico_rot"], ["no_rot", "full_rot"], [
            "ico_rot", "ico_rot"], ["ico_rot", "full_rot"], ["full_rot", "full_rot"]]
        for config in configs:
            # result
            print("Current configuration: training set: {}, test set: {}".format(
                config[0], config[1]))
            dataset = {}

            rot_type = {"train": config[0], "test": config[1]}

            for label, data in zip(["train", "test"], [mnist_train, mnist_test]):

                print("projecting {0} data set".format(label))
                current = 0
                signals = data['images'].reshape(-1, 28, 28).astype(np.float64)
                n_signals = signals.shape[0]

                if rot_type[label] in ['ico_rot', 'full_rot']:
                    n_rots = 60
                else:
                    n_rots = 1

                # array to store the projected MNIST digits in.
                projections = np.ndarray(
                    (signals.shape[0]*n_rots, grid.shape[0], grid.shape[1]), dtype=np.uint8)

                # if the rotation type is icosahedral we precompute a list of all charts so we don't have do do it
                # again in every step.
                if rot_type[label] == 'ico_rot':
                    perms = all_rotations_icosahedron()
                    # get object with correct shape to store values in
                    rotated_grids = np.zeros((n_rots, *grid.shape))

                    for i, perm in enumerate(perms):
                        ico_r = Icosahedron(r=4, rad=1,
                                            c=np.array([0, 0, 0]), perm=perm)
                        rotated_grids[i] = ico_r.get_charts_cut().reshape(
                            grid.shape)

                while current < n_signals:  # as long as we still have MNIST digits to process...
                    if rot_type[label] == 'no_rot':
                        # if no rotation just keep identity.
                        rotated_grids = grid[None, :]
                    elif rot_type[label] == 'full_rot':
                        # if rot type is full: get n_rots random rotation matrices:
                        rotated_grids = np.zeros((n_rots, *grid.shape))
                        for i in range(n_rots):
                            rot = rand_rotation_matrix(deflection=1.0)
                            rotated_grids[i] = ico.get_rotated_charts_cut(
                                rot).reshape(grid.shape)
                    # elif rot_type[label] == 'ico_rot':  # do nothing because we have already precomputed.

                    # get indices of the current chunk
                    idxs = np.arange(current, min(n_signals,
                                                  current + 500))
                    chunk = signals[idxs]  # get a chunk of MNIST
                    for i, r_grid in enumerate(rotated_grids):
                        projections[idxs*n_rots +
                                    i] = project_2d_on_sphere(chunk, r_grid.transpose(2, 0, 1))
                    current += args.chunk_size
                    print("\r{0}/{1}".format(current, n_signals), end="")
                print("")
                dataset[label] = {
                    'images': projections,
                    'labels': np.repeat(data['labels'], n_rots)
                }
            print("writing pickle")
            with gzip.open("MNIST_data/sph_ico_mnist_train_{}_test_{}.gz".format(config[0], config[1]), 'wb') as f:
                pickle.dump(dataset, f)
            print("done")
    else:
        # result
        dataset = {}

        rot_type = {"train": args.rot_type_train, "test": args.rot_type_test}

        for label, data in zip(["train", "test"], [mnist_train, mnist_test]):

            print("projecting {0} data set".format(label))
            current = 0
            signals = data['images'].reshape(-1, 28, 28).astype(np.float64)
            n_signals = signals.shape[0]

            if rot_type[label] in ['ico_rot', 'full_rot']:
                n_rots = 60
            else:
                n_rots = 1

            # array to store the projected MNIST digits in.
            projections = np.ndarray(
                (signals.shape[0]*n_rots, grid.shape[0], grid.shape[1]), dtype=np.uint8)

            # if the rotation type is icosahedral we precompute a list of all charts so we don't have do do it
            # again in every step.
            if rot_type[label] == 'ico_rot':
                perms = all_rotations_icosahedron()
                # get object with correct shape to store values in
                rotated_grids = np.zeros((n_rots, *grid.shape))

                for i, perm in enumerate(perms):
                    ico_r = Icosahedron(r=args.refinement, rad=1,
                                        c=np.array([0, 0, 0]), perm=perm)
                    rotated_grids[i] = ico_r.get_charts_cut().reshape(
                        grid.shape)

            while current < n_signals:  # as long as we still have MNIST digits to process...
                if rot_type[label] == 'no_rot':
                    # if no rotation just keep identity.
                    rotated_grids = grid[None, :]
                elif rot_type[label] == 'full_rot':
                    # if rot type is full: get n_rots random rotation matrices:
                    rotated_grids = np.zeros((n_rots, *grid.shape))
                    for i in range(n_rots):
                        rot = rand_rotation_matrix(deflection=args.noise)
                        rotated_grids[i] = ico.get_rotated_charts_cut(
                            rot).reshape(grid.shape)
                # elif rot_type[label] == 'ico_rot':  # do nothing because we have already precomputed.

                # get indices of the current chunk
                idxs = np.arange(current, min(n_signals,
                                              current + args.chunk_size))
                chunk = signals[idxs]  # get a chunk of MNIST
                for i, r_grid in enumerate(rotated_grids):
                    projections[idxs*n_rots +
                                i] = project_2d_on_sphere(chunk, r_grid.transpose(2, 0, 1))
                current += args.chunk_size
                print("\r{0}/{1}".format(current, n_signals), end="")
            print("")
            dataset[label] = {
                'images': projections,
                'labels': np.repeat(data['labels'], n_rots)
            }
        print("writing pickle")
        with gzip.open(args.output_file, 'wb') as f:
            pickle.dump(dataset, f)
        print("done")


if __name__ == '__main__':
    main()
