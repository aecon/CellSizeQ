import os
import sys
import glob
from skimage import io, measure
import numpy as np
import pandas as pd
from multiprocessing import Pool
import time
import ctypes
import numba
import argparse



@numba.jit(nopython=True, parallel=True, fastmath=True)
def get_counts(unique, counts, all_counts):
    n = len(unique)
    for i in numba.prange(n):
        l = unique[i]
        all_counts[l] = counts[i]


# Label detected nuclei the same colour as their corresponding cell
# If no corresponding cells exists, (with label>1), mark pixel with 0.
@numba.jit(nopython=True, parallel=True, fastmath=True)
def label_nuclei(image_cells, image_nuclei_seeds, image_nuclei):
    nx, ny = image_cells.shape
    for i in numba.prange(nx):
        for j in numba.prange(ny):
            l = image_cells[i,j]
            n = image_nuclei_seeds[i,j]
            image_nuclei[i,j] = l if (l>1 and n==1) else 0


def find_coordinates(Nlabels, coordinates, counts_cumsum, image):

    coordinates = coordinates.astype(np.uint64)
    counts_cumsum = counts_cumsum.astype(np.uint64)
    image_flat = image.flatten().astype(np.uint64)

    Nx,Ny = image.shape

    """
    coordinates = np.zeros((Nlabels, VolMax, 2))
    
    coordinates_flat:
        - size = Nlabels * VolMax * 2
        - flattened pattern: [[[x  y] per pixel] per object]
    
    array([[[ 11.,  12.],
        [ 22.,  24.],
        [ 33.,  36.]],

       [[ 21.,  22.],
        [ 42.,  44.],
        [ 63.,  66.]],

       [[ 31.,  32.],
        [ 62.,  64.],
        [ 93.,  96.]],

       [[ 41.,  42.],
        [ 82.,  84.],
        [123., 126.]]])

    >>> x.flatten()
    array([ 11.,  12.,  22.,  24.,  33.,  36.,  21.,  22.,  42.,  44.,  63.,
            66.,  31.,  32.,  62.,  64.,  93.,  96.,  41.,  42.,  82.,  84.,
           123., 126.])
    """


    # Calling the C function
    #
    path = os.path.dirname(os.path.realpath(__file__))
    lib = np.ctypeslib.load_library('objects.so', path)
    fun = lib.find_coordinates
    fun.restype = None

    # void find_coordinates(uint64_t Ni, uint64_t Nj, uint64_t Ncells,
    #                       uint64_t *coordinates, uint64_t *counts_cumsum, uint64_t *image)
    fun.argtypes = [
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        np.ctypeslib.ndpointer(np.dtype(np.uint64), flags='aligned, f_contiguous, writeable'),
        np.ctypeslib.ndpointer(np.dtype(np.uint64), flags='aligned, f_contiguous, writeable'),
        np.ctypeslib.ndpointer(np.dtype(np.uint64), flags='aligned, f_contiguous, writeable')
    ]
    fun(Nx, Ny, Nlabels,  coordinates, counts_cumsum, image_flat)

    return coordinates


def object_moments(Nlabels, coordinates, counts_cumsum, all_counts, moments):

    coordinates = coordinates.astype(np.uint64)
    counts_cumsum = counts_cumsum.astype(np.uint64)
    all_counts = all_counts.astype(np.uint64)
    moments_flatten = moments.flatten().astype(np.uint64)  # moments = np.zeros((Maxlabel+1, Nmoments), dtype=np.uint64)
    Nmoments = int(6)
    #print("before", np.sum(moments_flatten))

    # Calling the C function
    path = os.path.dirname(os.path.realpath(__file__))
    lib = np.ctypeslib.load_library('objects.so', path)
    fun = lib.object_moments
    fun.restype = None

    #void object_moments(const uint64_t Nlabels, const uint64_t Nmoments
    #                    uint64_t *coordinates, uint64_t *counts_cumsum, uint64_t *all_counts, float *moments)
    fun.argtypes = [
        ctypes.c_ulong,
        ctypes.c_ulong,
        np.ctypeslib.ndpointer(np.dtype(np.uint64), flags='aligned, f_contiguous, writeable'),
        np.ctypeslib.ndpointer(np.dtype(np.uint64), flags='aligned, f_contiguous, writeable'),
        np.ctypeslib.ndpointer(np.dtype(np.uint64), flags='aligned, f_contiguous, writeable'),
        np.ctypeslib.ndpointer(np.dtype(np.uint64), flags='aligned, f_contiguous, writeable')
    ]
    fun(Nlabels, Nmoments,  coordinates, counts_cumsum, all_counts, moments_flatten)

    shape = np.shape(moments)
    moments[:,:] = np.reshape(moments_flatten, shape)


# Possible race condition
#
#@numba.jit(nopython=True, parallel=True, fastmath=True)
#def object_moments(MaxLables, coordinates, all_counts, counts_cumsum, moments):
#    # Loop over labels (in parallel)
#    for l in numba.prange(2, MaxLables+1):  # exclude labels 0(bkg), 1(nuclei shape cells)
#
#        base = counts_cumsum[l-1] * 2
#        area = int(all_counts[l])
#
#        for j in range(area):
#            icx = int(base + j*2)
#            icy = int(icx + 1)
#            x = coordinates[icx]
#            y = coordinates[icy]

#            moments[l,0] += 1       # M00
#            moments[l,1] += x       # M10
#            moments[l,2] += y       # M01
#            moments[l,3] += x*y     # M11
#            moments[l,4] += x*x     # M20
#            moments[l,5] += y*y     # M02


@numba.jit(nopython=True, parallel=True, fastmath=True)
def object_transform(MaxLables, moments, counts_cumsum, coordinates, rotated_coordinates):

    # Loop over labels (in parallel)
    for l in numba.prange(2, MaxLables+1):  # exclude labels 0(bkg), 1(nuclei shape cells)

        # https://en.wikipedia.org/wiki/Image_moment
        M00 = float(moments[l,0])
        M10 = float(moments[l,1])
        M01 = float(moments[l,2])
        M11 = float(moments[l,3])
        M20 = float(moments[l,4])
        M02 = float(moments[l,5])

        # centroid
        xm = M10 / M00
        ym = M01 / M00

        # orientation
        mu11 = (M11 / M00) - xm*ym
        mu20 = (M20 / M00) - xm*xm
        mu02 = (M02 / M00) - ym*ym
        theta = 0.5 * np.arctan(2*mu11 / (mu20 - mu02))  # output in [-pi/2, pi/2]

        # Loop over coordinate to transform and measure axis lengths
        base = counts_cumsum[l-1] * 2
        area = M00
        for j in range(area):
            icx = int(base + j*2)
            icy = int(icx + 1)
            x = coordinates[icx]
            y = coordinates[icy]

            # centered coordinates
            xc = x-xm
            yc = y-ym

            # rotate by the opposite direction of theta
            # https://en.wikipedia.org/wiki/Rotation_matrix
            xr = xc*np.cos(-theta) - yc*np.sin(-theta)
            yr = xc*np.sin(-theta) + yc*np.cos(-theta)
            rotated_coordinates[icx] = xr
            rotated_coordinates[icy] = yr


@numba.jit(nopython=True, parallel=True, fastmath=True)
def object_lengths(MaxLables, counts_cumsum, all_counts, rotated_coordinates, lengths):

    # Loop over labels (in parallel)
    for l in numba.prange(2, MaxLables+1):  # exclude labels 0(bkg), 1(nuclei shape cells)

        # Find object's transformed coordinates
        base = counts_cumsum[l-1] * 2
        area = int(all_counts[l])

        if area > 0:
            icx0 = int(base)
            icy0 = int(icx0 + 1)
            icx1 = int(base + (area-1)*2)
            icy1 = int(icx1 + 1)

            X = rotated_coordinates[icx0:icx1:2]
            Y = rotated_coordinates[icy0:icy1:2]

            # Measure major and minor axis lengths
            length_1 = max(X)-min(X)  # major axis length
            length_2 = max(Y)-min(Y)  # minor axis length
            if length_1 < length_2:
                tmp = length_1
                length_1 = length_2
                length_2 = tmp

            lengths[l*2]   = length_1
            lengths[l*2+1] = length_2



def process_images_multithread(image_pair):
    cell_image_path, nucleus_image_path = image_pair

    # Load the cell and nucleus images
    image_cells = io.imread(cell_image_path, plugin='tifffile')
    image_nuclei_seeds = io.imread(nucleus_image_path, plugin='tifffile')
    #print(cell_image_path)


    # Get unique labels for cells (excluding the background)
    unique_cell_labels, counts_cells = np.unique(image_cells[image_cells>0], return_counts=True)
    Ncells = len(unique_cell_labels)
    Maxlabel = np.max(unique_cell_labels)

    # Get unique labels for nuclei, based on cell labels
    image_nuclei = np.zeros_like(image_cells)
    label_nuclei(image_cells, image_nuclei_seeds, image_nuclei)
    unique_nuclei_labels, counts_nuclei = np.unique(image_nuclei[image_nuclei>0], return_counts=True)

#    import matplotlib.pyplot as plt
#    plt.subplot(1,2,1)
#    plt.imshow(image_cells)
#    plt.subplot(1,2,2)
#    plt.imshow(image_nuclei)
#    plt.show()


    # Make complete label and counts arrays
    all_labels = np.arange(Maxlabel+1)

    all_counts_cells = np.zeros(Maxlabel+1)
    all_counts_nuclei = np.zeros(Maxlabel+1)
    
    get_counts(unique_cell_labels, counts_cells, all_counts_cells)
    get_counts(unique_nuclei_labels, counts_nuclei, all_counts_nuclei)

    counts_cumsum_cells  = np.cumsum(all_counts_cells).astype(np.uint64)
    counts_cumsum_nuclei = np.cumsum(all_counts_nuclei).astype(np.uint64)

    # Get pixel coordinates of each object
    coordinates_cells = np.zeros( int(counts_cumsum_cells[-1]*2) )
    coordinates_cells = find_coordinates(Maxlabel, coordinates_cells, counts_cumsum_cells, image_cells)

    coordinates_nuclei = np.zeros( int(counts_cumsum_nuclei[-1]*2) )
    coordinates_nuclei = find_coordinates(Maxlabel, coordinates_nuclei, counts_cumsum_nuclei, image_nuclei)

#    # Check coordinates - OK
#    for i in range(2,Maxlabel+1):
#        base = counts_cumsum_nuclei[i-1] * 2
#        vol = int(all_counts_nuclei[i])
#        print("label vol base", i, vol, base)
#        for j in range(vol):
#            icx = int(base + j*2)
#            icy = int(icx + 1);
#            print(coordinates_nuclei[icx], coordinates_nuclei[icy])


    # Calculate object moments
    Nmoments = 6   # number of moments to compute
    moments_cells  = np.zeros((Maxlabel+1, Nmoments), dtype=np.uint64)
    moments_nuclei = np.zeros((Maxlabel+1, Nmoments), dtype=np.uint64)

    #object_moments(Maxlabel, coordinates, all_counts, counts_cumsum, moments)  # numba version: possible race condition
    object_moments(Maxlabel, coordinates_cells,  counts_cumsum_cells,  all_counts_cells,  moments_cells)     # OpenMP version
    object_moments(Maxlabel, coordinates_nuclei, counts_cumsum_nuclei, all_counts_nuclei, moments_nuclei)    # OpenMP version

#    # Check M00 - OK
#    for i in range(0,15+1):
#        area = all_counts[i]
#        M00 = moments[i,0]
#        print("Cell %d, area: %d, M00: %d" % (i, area, M00))


    # Calculate object area, major and minor axis lengths
    # - transform object coordinates (center and rotate to principal axis)
    rotated_coordinates_cells  = np.zeros_like(coordinates_cells, dtype=np.float32)
    rotated_coordinates_nuclei = np.zeros_like(coordinates_nuclei, dtype=np.float32)

    object_transform(Maxlabel, moments_cells, counts_cumsum_cells, coordinates_cells, rotated_coordinates_cells)
    object_transform(Maxlabel, moments_nuclei, counts_cumsum_nuclei, coordinates_nuclei, rotated_coordinates_nuclei)

#    # Check transformed coordinates
#    import matplotlib.pyplot as plt
#    for i in range(2, 100):
#        base = int(counts_cumsum[i-1] * 2)
#        area = int(all_counts[i])
#        if area > 0:
#            print(i, area)
#
#            # Find cell's original and transformed coordinates
#            icx0 = int(base)
#            icy0 = int(icx0 + 1)
#            icx1 = int(base + (area-1)*2)
#            icy1 = int(icx1 + 1)
#            X  = coordinates[icx0:icx1:2]
#            Y  = coordinates[icy0:icy1:2]
#            XR = rotated_coordinates[icx0:icx1:2]
#            YR = rotated_coordinates[icy0:icy1:2]
#            #print(X, Y)
#            #print(XR, YR)
#
#            plt.scatter(X-np.mean(X), Y-np.mean(Y), alpha=0.4, label="original")
#            plt.scatter(XR, YR, alpha=0.4, label="rotated")
#            plt.gca().set_aspect('equal', 'box')
#            plt.tight_layout()
#            plt.legend()
#            plt.show()


    # - measure axis lengths
    axis_lengths_cells  = np.zeros(((Maxlabel+1)*2), dtype=np.float32)
    axis_lengths_nuclei = np.zeros(((Maxlabel+1)*2), dtype=np.float32)

    object_lengths(Maxlabel, counts_cumsum_cells,  all_counts_cells,  rotated_coordinates_cells,  axis_lengths_cells)
    object_lengths(Maxlabel, counts_cumsum_nuclei, all_counts_nuclei, rotated_coordinates_nuclei, axis_lengths_nuclei)

#    # Check long and short lengths
#    import matplotlib.pyplot as plt
#    for i in range(2, 100):
#        base = int(counts_cumsum[i-1] * 2)
#        area = int(all_counts[i])
#        if area > 0:
#            print(i, area)
#
#            # Find cell's original and transformed coordinates
#            icx0 = int(base)
#            icy0 = int(icx0 + 1)
#            icx1 = int(base + (area-1)*2)
#            icy1 = int(icx1 + 1)
#            XR = rotated_coordinates[icx0:icx1:2]
#            YR = rotated_coordinates[icy0:icy1:2]
#
#            major_length = axis_lengths[i*2]
#            minor_length = axis_lengths[i*2+1]
#
#            plt.scatter(XR, YR, label="rotated")
#            plt.gca().set_aspect('equal', 'box')
#            plt.title("M1:%.1f, L2:%.1f" % (major_length, minor_length))
#            plt.tight_layout()
#            plt.legend()
#            plt.show()



    # Compose object dictionaries
    cell_data = []
    nucleus_data = []  # TODO

    for i in range(2, Maxlabel+1):
        area_cells = int(all_counts_cells[i])
        area_nuclei = int(all_counts_nuclei[i])
        
        if area_cells > 0:
            #print(i, area)

            major_length_cells = axis_lengths_cells[i*2]
            minor_length_cells = axis_lengths_cells[i*2+1]
            major_length_nuclei = axis_lengths_nuclei[i*2]
            minor_length_nuclei = axis_lengths_nuclei[i*2+1]

            # append to lists
            cell_data.append({
                'Label': i,
                'Area': area_cells,
                'Length': major_length_cells,
                'Width': minor_length_cells
            })
            nucleus_data.append({
                'Label': i,
                'Area': area_nuclei,
                'Length': major_length_nuclei,
                'Width': minor_length_nuclei
            })

    return cell_data , nucleus_data


# Multiprocessing version using python libraries
# (scikit-image) to compute shape properties.
# 
# Too slow. Do not use.
#def process_images_multiprocessing(image_pair):
#    cell_image_path, nucleus_image_path = image_pair
#
#    # Load the cell and nucleus images
#    image_cells = io.imread(cell_image_path, plugin='tifffile')
#    image_nuclei = io.imread(nucleus_image_path, plugin='tifffile')
#    print(cell_image_path)
#
#    # Get unique labels for cells (excluding the background)
#    unique_cell_labels, counts_cells = np.unique(image_cells, return_counts=True)
#    unique_cell_labels = unique_cell_labels[unique_cell_labels > 1]  # Exclude background
#
#    cell_data = []
#    nucleus_data = []
#
#    # Calculate properties for each cell and nucleus
#    nuclei_labels = image_cells.copy()
#    nuclei_labels[image_nuclei<1] = 0
#
#    global_index = 0
#    for cell_label in unique_cell_labels:
#        # pixel mask of particular cell and correponding nucleus
#        cell_mask = np.zeros_like(image_cells)
#        cell_mask[image_cells == cell_label] = 1
#        nucleus_mask = np.zeros_like(image_cells)
#        nucleus_mask[nuclei_labels == cell_label] = 1
#
#        area_nucleus = np.sum(nucleus_mask)
#        area_cell = np.sum(cell_mask)
#        #print(area_nucleus, area_cell)
#
#        # compute region properties
#        cell_region = measure.regionprops(cell_mask.astype(int))[0]  # Assuming one region per label
#        cell_area = cell_region.area
#        cell_major_axis_length = cell_region.major_axis_length
#        cell_minor_axis_length = cell_region.minor_axis_length
#
#        if area_nucleus > 1:
#            nucleus_region = measure.regionprops(nucleus_mask.astype(int))[0]  # Assuming one region per label
#            nucleus_area = nucleus_region.area
#            nucleus_major_axis_length = nucleus_region.major_axis_length
#            nucleus_minor_axis_length = nucleus_region.minor_axis_length
#        else:
#            nucleus_area = 0
#            nucleus_major_axis_length = 0
#            nucleus_minor_axis_length = 0
#        #print(nucleus_area, cell_area)
#
#        # append to lists
#        cell_data.append({
#            'Label': global_index,
#            'Area': cell_area,
#            'Length': cell_major_axis_length,
#            'Width': cell_minor_axis_length
#        })
#        nucleus_data.append({
#            'Label': global_index,
#            'Area': nucleus_area,
#            'Length': nucleus_major_axis_length,
#            'Width': nucleus_minor_axis_length
#        })
#
#        global_index = global_index + 1
#
#
#    return cell_data, nucleus_data


def check_image_numbers(paths_cells, paths_nuclei):
    # check number of images
    Nimg_cells = len(paths_cells)
    Nimg_nuclei = len(paths_nuclei)
    assert(Nimg_cells > 0)
    assert(Nimg_cells == Nimg_nuclei)
    Nimg = Nimg_cells
    print("Found %d images." % Nimg)


def main(input_dir_cells, input_dir_nuclei, cell_output_csv, nucleus_output_csv, plate_name):

    # Gather paths for cell and nucleus images
    cell_image_paths  = sorted( glob.glob("%s/*_cell_labels.tif" % input_dir_cells) )
    nucleus_image_paths = sorted( glob.glob("%s/*_seeds.tif" % input_dir_nuclei) )
    #nucleus_image_paths = sorted( glob.glob("%s/*_cell_labels.tif" % input_dir_nuclei) )
    check_image_numbers(cell_image_paths, nucleus_image_paths)

    # Create pairs of cell and nucleus image paths
    image_pairs = list(zip(cell_image_paths, nucleus_image_paths))

    #if 0:
    #    # Use multiprocessing to process images in parallel
    #    max_processes = 8
    #    print("Starting multiprocessing with %d processes." % max_processes)
    #    with Pool(processes=max_processes) as pool:
    #        results = pool.map(process_images_multiprocessing, image_pairs)
    if 1:
        # Process single image pair
        # Parallelizetion over pixels and objects
        results = []
        for inum, pair in enumerate(image_pairs):
            t0 = time.time()
            r = process_images_multithread(pair)
            t1 = time.time()
            if inum%20 == 0:
                print(pair[0])
                print("Time for one image:", t1-t0)
            results.append(r)


    # Prepare DataFrames for cells and nuclei
    cell_df_list = []
    nucleus_df_list = []

    for i, (cells, nuclei) in enumerate(results):

        # Extract image ID from the file name (assuming both images have the same base name)
        image_id_cells = os.path.basename(image_pairs[i][0]) #.replace('_cell_labels.tif', '')
        image_id_nuclei = os.path.basename(image_pairs[i][1]) #.replace('_seeds.tif', '')

        # Store plate name, row, column, and field in the dictionaries
        row = image_id_cells.split(' - ')[0][-1]
        column = image_id_cells.split(' - ')[1].split('(fld')[0]
        field = image_id_cells.split('fld ')[1].split(' wv')[0]

        # -cell data
        for cell in cells:
            cell_df_list.append({**cell,
                                 'Plate': plate_name,
                                 'Row': row,
                                 'Column': column,
                                 'Field': field,
                                 'Image': image_id_cells
                                 })

        # -nucleus data
        for nucleus in nuclei:
            nucleus_df_list.append({**nucleus,
                                 'Plate': plate_name,
                                 'Row': row,
                                 'Column': column,
                                 'Field': field,
                                 'Image': image_id_nuclei
                                 })


    # Create DataFrames
    cell_df = pd.DataFrame(cell_df_list)
    nucleus_df = pd.DataFrame(nucleus_df_list)

    # Save to CSV files
    cell_df.to_csv(cell_output_csv, index=False)
    nucleus_df.to_csv(nucleus_output_csv, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str, required=True, help="plate name")
    parser.add_argument('-d', type=str, required=True, help="directory containing cells and nuclei subdirs")
    args = parser.parse_args()

    plate_name = "HA05_rep1"

    # DIRECTORIES
    image_folder = args.d
    plate_name   = args.n
    input_dir_cells  = image_folder + os.sep + "cells"
    input_dir_nuclei = image_folder + os.sep + "nuclei"

    # OUTPUT FILES
    cell_output_csv  = "results" + os.sep + "out_%s_cells_size.csv" % plate_name
    nucleus_output_csv = "results" + os.sep + "out_%s_nuclei_size.csv" % plate_name

    main(input_dir_cells, input_dir_nuclei, cell_output_csv, nucleus_output_csv, plate_name)

