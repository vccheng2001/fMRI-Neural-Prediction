import pandas as pd
import numpy as np
import csv 
import sys 
import matplotlib.pyplot as plt
import imagesc as imagesc
import scipy.io as c
import pylab as pl                    

# Visualize 2D fMRI brain slices using images from training data 
# python3 visualize.py <path/to/data> <row_index> 
def main():
    num_voxels = 21764
    nx, ny, nz = 51, 61, 23

    # maps voxel index -> xyz coordinates
    col_to_coord = np.genfromtxt("col_to_coord.csv", delimiter=",")
    # stores all voxel values
    data = np.genfromtxt(dataset, delimiter=",", skip_header=1,  dtype='unicode')

    # store xyz coords of each voxel 
    coords = {}
    for i in range(num_voxels):
        coords[i] = col_to_coord[i]

    label = data[row_index,-1]            # class category
    voxel_values = data[row_index,:-1]    # feature values

    # construct 3d representation of image 
    img_3d = np.empty((nx,ny,nz))
    for i in range(num_voxels):
        x = int(coords[i][0])
        y = int(coords[i][1])
        z = int(coords[i][2])
        img_3d[x, y, z] = voxel_values[i]

    # plot slice by slice 
    pl.ion()
    for z in range(nz):
        pl.cla()
        pl.imshow(img_3d[:,:,z], interpolation='nearest')
        pl.draw()
        pl.xlabel("x")
        pl.ylabel("y")
        pl.title(label)
        pl.pause(0.3)
    pl.ioff()


if __name__ == "__main__":
    # takes in row number from training data
    (program, dataset, row_index) = sys.argv 
    row_index = int(row_index)
    main()