################################################################################
# COMP3317 Computer Vision
# Assignment 4 - Camera calibration
################################################################################
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import convolve1d
from numpy.linalg import lstsq, qr, inv

################################################################################
#  estimate planar projective transformations for the 2 calibration planes
################################################################################
def calibrate2D(ref3D, ref2D) :
    #  input:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D
    # return:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    Hxz = np.zeros(shape=(3,3), dtype = np.float64)
    Hyz = np.zeros(shape=(3,3), dtype = np.float64)
    points3DXZ = ref3D[0:4]
    points2DXZ = ref2D[0:4]
    points3DYZ = ref3D[4:]
    points2DYZ = ref2D[4:]

    # TODO : form the matrix equation Ap = b for the X-Z plane

    Axz = np.zeros(shape=(8,8), dtype = np.float64)
    Bxz = np.zeros(shape=(8,1), dtype = np.float64)

    for index in range(4):
        xi = points3DXZ[index][0] # world x
        yi = points3DXZ[index][2] # world y
        ui = points2DXZ[index][0]  # image x
        vi = points2DXZ[index][1] # image y

        firstRowIndex = index * 2
        secondRowIndex = index * 2 + 1
        Axz[firstRowIndex][0] = xi
        Axz[firstRowIndex][1] = yi
        Axz[firstRowIndex][2] = 1
        Axz[firstRowIndex][6] = -ui * xi
        Axz[firstRowIndex][7] = -ui * yi

        Axz[secondRowIndex][3] = xi
        Axz[secondRowIndex][4] = yi
        Axz[secondRowIndex][5] = 1
        Axz[secondRowIndex][6] = -vi * xi
        Axz[secondRowIndex][7] = -vi * yi

        Bxz[firstRowIndex][0] = ui
        Bxz[secondRowIndex][0] = vi

    # TODO : solve for the planar projective transformation using linear least squares

    # (8,1) shape
    Pxz = np.linalg.lstsq(Axz, Bxz)[0]
    count = 0
    for row in range(3):
        for col in range(3):
            if row == 2 and col == 2:
                Hxz[row][col] = 1
            else:
                Hxz[row][col] = Pxz[count]
            count += 1

    # TODO : form the matrix equation Ap = b for the Y-Z plane

    Ayz = np.zeros(shape=(8,8), dtype = np.float64)
    Byz = np.zeros(shape=(8,1), dtype = np.float64)

    for index in range(4):
        xi = points3DYZ[index][1] # world x
        yi = points3DYZ[index][2] # world y
        ui = points2DYZ[index][0]  # image x
        vi = points2DYZ[index][1] # image y

        firstRowIndex = index * 2
        secondRowIndex = index * 2 + 1
        Ayz[firstRowIndex][0] = xi
        Ayz[firstRowIndex][1] = yi
        Ayz[firstRowIndex][2] = 1
        Ayz[firstRowIndex][6] = -ui * xi
        Ayz[firstRowIndex][7] = -ui * yi

        Ayz[secondRowIndex][3] = xi
        Ayz[secondRowIndex][4] = yi
        Ayz[secondRowIndex][5] = 1
        Ayz[secondRowIndex][6] = -vi * xi
        Ayz[secondRowIndex][7] = -vi * yi

        Byz[firstRowIndex][0] = ui
        Byz[secondRowIndex][0] = vi
    # TODO : solve for the planar projective transformation using linear least squares

    # (8,1) shape
    Pyz = np.linalg.lstsq(Ayz, Byz)[0]
    count = 0
    for row in range(3):
        for col in range(3):
            if row == 2 and col == 2:
                Hyz[row][col] = 1
            else:
                Hyz[row][col] = Pyz[count]
            count += 1

    return Hxz, Hyz

################################################################################
#  generate correspondences for all the corners on the 2 calibration planes
################################################################################
def gen_correspondences(Hxz, Hyz, corners) :
    # input:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane
    #    corners - a n0 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n0 being the number of corners)
    # return:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D

    # TODO : define 3D coordinates of all the corners on the 2 calibration planes
    corners3DXZ = np.array([[0.5, 0, 0.5], [1.5, 0, 0.5], [0.5, 0, 1.5], [1.5, 0, 1.5],
                           [2.5, 0, 0.5], [3.5, 0, 0.5], [2.5, 0, 1.5], [3.5, 0, 1.5],
                           [4.5, 0, 0.5], [5.5, 0, 0.5], [4.5, 0, 1.5], [5.5, 0, 1.5],
                           [6.5, 0, 0.5], [7.5, 0, 0.5], [6.5, 0, 1.5], [7.5, 0, 1.5],
                           [8.5, 0, 0.5], [9.5, 0, 0.5], [8.5, 0, 1.5], [9.5, 0, 1.5],
                           [0.5, 0, 2.5], [1.5, 0, 2.5], [0.5, 0, 3.5], [1.5, 0, 3.5],
                           [2.5, 0, 2.5], [3.5, 0, 2.5], [2.5, 0, 3.5], [3.5, 0, 3.5],
                           [4.5, 0, 2.5], [5.5, 0, 2.5], [4.5, 0, 3.5], [5.5, 0, 3.5],
                           [6.5, 0, 2.5], [7.5, 0, 2.5], [6.5, 0, 3.5], [7.5, 0, 3.5],
                           [8.5, 0, 2.5], [9.5, 0, 2.5], [8.5, 0, 3.5], [9.5, 0, 3.5],
                           [0.5, 0, 4.5], [1.5, 0, 4.5], [0.5, 0, 5.5], [1.5, 0, 5.5],
                           [2.5, 0, 4.5], [3.5, 0, 4.5], [2.5, 0, 5.5], [3.5, 0, 5.5],
                           [4.5, 0, 4.5], [5.5, 0, 4.5], [4.5, 0, 5.5], [5.5, 0, 5.5],
                           [6.5, 0, 4.5], [7.5, 0, 4.5], [6.5, 0, 5.5], [7.5, 0, 5.5],
                           [8.5, 0, 4.5], [9.5, 0, 4.5], [8.5, 0, 5.5], [9.5, 0, 5.5],
                           [0.5, 0, 6.5], [1.5, 0, 6.5], [0.5, 0, 7.5], [1.5, 0, 7.5],
                           [2.5, 0, 6.5], [3.5, 0, 6.5], [2.5, 0, 7.5], [3.5, 0, 7.5],
                           [4.5, 0, 6.5], [5.5, 0, 6.5], [4.5, 0, 7.5], [5.5, 0, 7.5],
                           [6.5, 0, 6.5], [7.5, 0, 6.5], [6.5, 0, 7.5], [7.5, 0, 7.5],
                           [8.5, 0, 6.5], [9.5, 0, 6.5], [8.5, 0, 7.5], [9.5, 0, 7.5]]
                           )
    corners3DYZ = np.array([[0, 0.5, 0.5], [0, 1.5, 0.5], [0, 0.5, 1.5], [0, 1.5, 1.5],
                           [0, 2.5, 0.5], [0, 3.5, 0.5], [0, 2.5, 1.5], [0, 3.5, 1.5],
                           [0, 4.5, 0.5], [0, 5.5, 0.5], [0, 4.5, 1.5], [0, 5.5, 1.5],
                           [0, 6.5, 0.5], [0, 7.5, 0.5], [0, 6.5, 1.5], [0, 7.5, 1.5],
                           [0, 8.5, 0.5], [0, 9.5, 0.5], [0, 8.5, 1.5], [0, 9.5, 1.5],
                           [0, 0.5, 2.5], [0, 1.5, 2.5], [0, 0.5, 3.5], [0, 1.5, 3.5],
                           [0, 2.5, 2.5], [0, 3.5, 2.5], [0, 2.5, 3.5], [0, 3.5, 3.5],
                           [0, 4.5, 2.5], [0, 5.5, 2.5], [0, 4.5, 3.5], [0, 5.5, 3.5],
                           [0, 6.5, 2.5], [0, 7.5, 2.5], [0, 6.5, 3.5], [0, 7.5, 3.5],
                           [0, 8.5, 2.5], [0, 9.5, 2.5], [0, 8.5, 3.5], [0, 9.5, 3.5],
                           [0, 0.5, 4.5], [0, 1.5, 4.5], [0, 0.5, 5.5], [0, 1.5, 5.5],
                           [0, 2.5, 4.5], [0, 3.5, 4.5], [0, 2.5, 5.5], [0, 3.5, 5.5],
                           [0, 4.5, 4.5], [0, 5.5, 4.5], [0, 4.5, 5.5], [0, 5.5, 5.5],
                           [0, 6.5, 4.5], [0, 7.5, 4.5], [0, 6.5, 5.5], [0, 7.5, 5.5],
                           [0, 8.5, 4.5], [0, 9.5, 4.5], [0, 8.5, 5.5], [0, 9.5, 5.5],
                           [0, 0.5, 6.5], [0, 1.5, 6.5], [0, 0.5, 7.5], [0, 1.5, 7.5],
                           [0, 2.5, 6.5], [0, 3.5, 6.5], [0, 2.5, 7.5], [0, 3.5, 7.5],
                           [0, 4.5, 6.5], [0, 5.5, 6.5], [0, 4.5, 7.5], [0, 5.5, 7.5],
                           [0, 6.5, 6.5], [0, 7.5, 6.5], [0, 6.5, 7.5], [0, 7.5, 7.5],
                           [0, 8.5, 6.5], [0, 9.5, 6.5], [0, 8.5, 7.5], [0, 9.5, 7.5]]
                           )

    corners3D = []
    for corner in corners3DXZ:
        corners3D.append([corner[0], corner[2], 1.0])

    """
    corners3D = np.array([[0.5, 0.5, 1], [1.5, 0.5, 1], [0.5, 1.5, 1], [1.5, 1.5, 1],
                           [2.5, 0.5, 1], [3.5, 0.5, 1], [2.5, 1.5, 1], [3.5, 1.5, 1],
                           [4.5, 0.5, 1], [5.5, 0.5, 1], [4.5, 1.5, 1], [5.5, 1.5, 1],
                           [6.5, 0.5, 1], [7.5, 0.5, 1], [6.5, 1.5, 1], [7.5, 1.5, 1],
                           [8.5, 0.5, 1], [9.5, 0.5, 1], [8.5, 1.5, 1], [9.5, 1.5, 1],
                           [0.5, 2.5, 1], [1.5, 2.5, 1], [0.5, 3.5, 1], [1.5, 3.5, 1],
                           [2.5, 2.5, 1], [3.5, 2.5, 1], [2.5, 3.5, 1], [3.5, 3.5, 1],
                           [4.5, 2.5, 1], [5.5, 2.5, 1], [4.5, 3.5, 1], [5.5, 3.5, 1],
                           [6.5, 2.5, 1], [7.5, 2.5, 1], [6.5, 3.5, 1], [7.5, 3.5, 1],
                           [8.5, 2.5, 1], [9.5, 2.5, 1], [8.5, 3.5, 1], [9.5, 3.5, 1],
                           [0.5, 4.5, 1], [1.5, 4.5, 1], [0.5, 5.5, 1], [1.5, 5.5, 1],
                           [2.5, 4.5, 1], [3.5, 4.5, 1], [2.5, 5.5, 1], [3.5, 5.5, 1],
                           [4.5, 4.5, 1], [5.5, 4.5, 1], [4.5, 5.5, 1], [5.5, 5.5, 1],
                           [6.5, 4.5, 1], [7.5, 4.5, 1], [6.5, 5.5, 1], [7.5, 5.5, 1],
                           [8.5, 4.5, 1], [9.5, 4.5, 1], [8.5, 5.5, 1], [9.5, 5.5, 1],
                           [0.5, 6.5, 1], [1.5, 6.5, 1], [0.5, 7.5, 1], [1.5, 7.5, 1],
                           [2.5, 6.5, 1], [3.5, 6.5, 1], [2.5, 7.5, 1], [3.5, 7.5, 1],
                           [4.5, 6.5, 1], [5.5, 6.5, 1], [4.5, 7.5, 1], [5.5, 7.5, 1],
                           [6.5, 6.5, 1], [7.5, 6.5, 1], [6.5, 7.5, 1], [7.5, 7.5, 1],
                           [8.5, 6.5, 1], [9.5, 6.5, 1], [8.5, 7.5, 1], [9.5, 7.5, 1]]
                           )
    """

    # TODO : project corners on the calibration plane 1 onto the image

    corners2D = []
    for corner in corners3D:
        corners2D.append(np.matmul(Hxz, corner))

    # TODO : project corners on the calibration plane 2 onto the image
    for corner in corners3D:
        corners2D.append(np.matmul(Hyz, corner))
    # TODO : locate the nearest detected corners

    realCorners2D = []
    for corner in corners2D:
        realCorners2D.append([corner[0] / corner[2], corner[1] / corner[2]])
    npCorners2D = np.array(realCorners2D)

    ref3D = np.concatenate((corners3DXZ, corners3DYZ))
    ref2D = find_nearest_corner(npCorners2D, corners)

    return ref3D, ref2D

################################################################################
#  estimate the camera projection matrix
################################################################################
def calibrate3D(ref3D, ref2D) :
    # input:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D
    # output:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    # TODO : form the matrix equation Ap = b for the camera
    # picking 3 points in XZ plane, 3 points in YZ plane
    calibPoints3D = []
    calibPoints2D = []

    """
    Most extreme corners
    please click on the image point for (9.5, 0.0, 7.5)...
    please click on the image point for (0.5, 0.0, 7.5)...
    please click on the image point for (9.5, 0.0, 0.5)...
    
    please click on the image point for (0.0, 0.5, 7.5)...
    please click on the image point for (0.0, 9.5, 7.5)...
    please click on the image point for (0.0, 9.5, 0.5)...
    
    np.any(np.any(np.all(ref3D[i] == [9.5, 0.0, 7.5]), np.all(ref3D[i] == [0.5, 0.0, 7.5]), np.all(ref3D[i] == [9.5, 0.0, 0.5])),
            np.all(ref3D[i] == [0.0, 0.5, 7.5]), np.all(ref3D[i] == [0.0, 9.5, 7.5]), np.all(ref3D[i] == [0.0, 9.5, 0.5])):
    """

    for i in range(160):
        if np.all(ref3D[i] == [9.5, 0.0, 7.5]) or np.all(ref3D[i] == [9.5, 0.0, 0.5]) or np.all(ref3D[i] == [0.5, 0.0, 7.5]) \
                or np.all(ref3D[i] == [0.0, 9.5, 7.5]) or np.all(ref3D[i] == [0.0, 9.5, 0.5]) or np.all(ref3D[i] == [0.0, 0.5, 0.5]):
            calibPoints3D.append(ref3D[i])
            calibPoints2D.append(ref2D[i])

    A = np.zeros(shape=(12,12), dtype = np.float64)

    for a in range(6):
        xi = calibPoints3D[a][0] # world x
        yi = calibPoints3D[a][1] # world y
        zi = calibPoints3D[a][2] # world z

        ui = calibPoints2D[a][0] # image x
        vi = calibPoints2D[a][1] # image y

        firstRowIndex = a * 2
        secondRowIndex = a * 2 + 1

        A[firstRowIndex][0] = xi
        A[firstRowIndex][1] = yi
        A[firstRowIndex][2] = zi
        A[firstRowIndex][3] = 1
        A[firstRowIndex][8] = -ui * xi
        A[firstRowIndex][9] = -ui * yi
        A[firstRowIndex][10] = -ui * zi
        A[firstRowIndex][11] = -ui

        A[secondRowIndex][4] = xi
        A[secondRowIndex][5] = yi
        A[secondRowIndex][6] = zi
        A[secondRowIndex][7] = 1
        A[secondRowIndex][8] = -vi * xi
        A[secondRowIndex][9] = -vi * yi
        A[secondRowIndex][10] = -vi * zi
        A[secondRowIndex][11] = -vi

    # TODO : solve for the projection matrix using linear least squares
    u, s, vh = np.linalg.svd(A)
    vhTransposed = vh.T
    P = np.zeros(shape=(3,4), dtype = np.float64)

    count = 0
    for row in range(3):
        for col in range(4):
            P[row][col] = vhTransposed[count][-1]
            count += 1
    return P

################################################################################
#  decompose the camera calibration matrix in K[R T]
################################################################################
def decompose_P(P) :
    # input:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix
    # output:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    # TODO: extract the 3 x 3 submatrix from the first 3 columns of P
    submat = P[:,:3]

    # TODO : perform QR decomposition on the inverse of [P0 P1 P2]

    q, r = np.linalg.qr(np.linalg.inv(submat))

    # TODO : obtain K as the inverse of R

    K = np.linalg.inv(r)

    # TODO : obtain R as the tranpose of Q

    R = q.T

    # TODO : normalize K
    alpha = 1

    if K[2][2] != 1:
        alpha = K[2][2]
        K /= alpha

    if K[0][0] < 0:
        for index in range(3):
            K[index][0] = -K[index][0]
            R[0][index] = -R[0][index]

    if K[1][1] < 0:
        for index in range(3):
            K[index][1] = -K[index][1]
            R[1][index] = -R[1][index]

    # TODO : obtain T from P3
    t = (1 / alpha) * np.matmul(np.linalg.inv(K), P[:, 3])
    T = np.reshape(t, (3, 1))

    RT = np.append(R, T, axis = 1)

    return K, RT

################################################################################
#  check the planar projective transformations for the 2 calibration planes
################################################################################
def check_H(img_color, Hxz, Hyz) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # define 3D coordinates of all the corners on the 2 calibration planes
    X_ = np.arange(10) + 0.5 # Y == X
    Z_ = np.arange(8) + 0.5
    X_ = np.tile(X_, 8)
    Z_ = np.repeat(Z_, 10)
    X = np.vstack((X_, Z_, np.ones(80)))

    # project corners on the calibration plane 1 onto the image
    w = Hxz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    # project corners on the calibration plane 2 onto the image
    w = Hyz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the 2D correspondences for the 2 calibration planes
################################################################################
def check_correspondences(img_color, ref2D) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners on the 2 calibration planes

    # plot the image and the correspondences
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(ref2D[:, 0], ref2D[:, 1], 'bx', markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the estimated camera projection matrix
################################################################################
def check_P(img_color, ref3D, P) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # project the reference 3D points onto the image
    w = P @ np.append(ref3D, np.ones([len(ref3D), 1]), axis = 1).T
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  pick seed corners for camera calibration
################################################################################
def pick_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)
    # return:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D

    # plot the image and corners
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(corners[:,0], corners[:,1],'r+', markersize = 5)
    plt.show()

    # define 3D coordinates of the extreme corners on the 2 calibration planes
    ref3D = np.array([(9.5, 0, 7.5), (0.5, 0, 7.5), (9.5, 0, 0.5), (0.5, 0, 0.5),
                      (0, 0.5, 7.5), (0, 9.5, 7.5), (0, 0.5, 0.5), (0, 9.5, 0.5)],
                      dtype = np.float64)
    ref2D = np.zeros([8, 2], dtype = np.float64)
    for i in range(8) :
        selected = False
        while not selected :
            # ask user to pick the corner on the image
            print('please click on the image point for ({}, {}, {})...'.format(
                  ref3D[i, 0], ref3D[i, 1], ref3D[i, 2]))
            plt.figure(fig.number)
            pt = plt.ginput(n = 1, timeout = - 1)
            # locate the nearest detected corner
            pt = find_nearest_corner(np.array(pt), corners)
            if pt[0, 0] > 0 :
                plt.figure(fig.number)
                plt.plot(pt[:, 0], pt[:, 1], 'bx', markersize = 5)
                ref2D[i, :] = pt[0]
                selected = True
            else :
                print('cannot locate detected corner in the vicinity...')
    plt.close(fig)

    return ref3D, ref2D

################################################################################
#  find nearest corner
################################################################################
def find_nearest_corner(pts, corners) :
    # input:
    #    pts - a n0 x 2 numpy ndarray holding the coordinates of the points
    #          (n0 being the number of points)
    #    corners - a n1 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n1 being the number of corners)
    # return:
    #    selected - a n0 x 2 numpy ndarray holding the coordinates of the nearest_corner
    #               corner

    x = corners[:, 0]
    y = corners[:, 1]
    x_ = pts[:, 0]
    y_ = pts[:, 1]

    # compute distances between the corners and the pts
    dist = np.sqrt(np.square(x.reshape(-1,1).repeat(len(x_), axis = 1) - x_)
                 + np.square(y.reshape(-1,1).repeat(len(y_), axis = 1) - y_))
    # find the index of the corner with the min distance for each pt
    min_idx = np.argmin(dist, axis = 0)
    # find the min distance for each pt
    min_dist = dist[min_idx, np.arange(len(x_))]
    # extract the nearest corner for each pt
    selected = corners[min_idx, 0:2]
    # identify corners with a min distance > 10 and replace them with [-1, -1]
    idx = np.where(min_dist > 10)
    selected[idx, :] = [-1 , -1]
    return selected

################################################################################
#  save K[R T] to a file
################################################################################
def save_KRT(outputfile, K, RT) :
    # input:
    #    outputfile - path of the output file
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(outputfile, 'w')
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(K[i,0], K[i, 1], K[i, 2]))
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e} {:.6e}\n'.format(RT[i, 0], RT[i, 1],
                       RT[i, 2], RT[i, 3]))
        file.close()
    except :
        print('Error occurs in writting output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load K[R T] from a file
################################################################################
def load_KRT(inputfile) :
    # input:
    #    inputfile - path of the file containing K[R T]
    # return:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(inputfile, 'r')
        K = np.zeros([3, 3], dtype = np.float64)
        RT = np.zeros([3, 4], dtype = np.float64)
        for i in range(3) :
            line = file.readline()
            e0, e1, e2 = line.split()
            K[i] = [np.float64(e0), np.float64(e1), np.float64(e2)]
        for i in range(3) :
            line = file.readline()
            e0, e1, e2, e3 = line.split()
            RT[i] = [np.float64(e0), np.float64(e1), np.float64(e2), np.float64(e3)]
        file.close()
    except :
        print('Error occurs in loading K[R T] from \'{}\'.'.format(inputfile))
        sys.exit(1)

    return K, RT

################################################################################
#  load image from a file
################################################################################
def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  load corners from a file
################################################################################
def load_corners(inputfile) :
    # input:
    #    inputfile - path of the file containing corner detection output
    # return:
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        # print('loading {} corners'.format(nc))
        corners = np.zeros([nc, 3], dtype = np.float64)
        for i in range(nc) :
            line = file.readline()
            x, y, r = line.split()
            corners[i] = [np.float64(x), np.float64(y), np.float64(r)]
        file.close()
        return corners
    except :
        print('Error occurs in loading corners from \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 4')
    parser.add_argument('-i', '--image', type = str, default = 'grid1.jpg',
                        help = 'filename of input image')
    # REVERT BACK LATER -> GRID1.CRN.TXT SHOULD BE GRID1.CRN
    parser.add_argument('-c', '--corners', type = str, default = 'grid1.crn',
                        help = 'filename of corner detection output')
    parser.add_argument('-o', '--output', type = str,
                        help = 'filename for outputting camera calibration result')
    args = parser.parse_args()

    print('-------------------------------------------')
    print('COMP3317 Assignment 4 - Camera calibration')
    print('input image : {}'.format(args.image))
    print('corner list : {}'.format(args.corners))
    print('output file : {}'.format(args.output))
    print('-------------------------------------------')

    # load the image
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))

    # load the corner detection result
    corners = load_corners(args.corners)
    print('{} corners loaded from \'{}\'...'.format(len(corners), args.corners))

    # pick the seed corners for camera calibration
    print('pick seed corners for camera calibration...')
    ref3D, ref2D = pick_corners(img_color, corners)

    # estimate planar projective transformations for the 2 calibration planes
    print('estimate planar projective transformations for the 2 calibration planes...')
    H1, H2 = calibrate2D(ref3D, ref2D)
    check_H(img_color, H1, H2)

    # generate correspondences for all the corners on the 2 calibration planes
    print('generate correspondences for all the corners on the 2 calibration planes...')
    ref3D, ref2D = gen_correspondences(H1, H2, corners)
    check_correspondences(img_color, ref2D)

    # estimate the camera projection matrix
    print('estimate the camera projection matrix...')
    P = calibrate3D(ref3D, ref2D)
    print('P = ')
    print(P)
    check_P(img_color, ref3D, P)

    # decompose the camera projection matrix into K[R T]
    print('decompose the camera projection matrix...')
    K, RT = decompose_P(P)
    print('K =')
    print(K)
    print('[R T] =')
    print(RT)
    check_P(img_color, ref3D, K @ RT)

    # save K[R T] to a file
    if args.output :
        save_KRT(args.output, K, RT)
        print('K[R T] saved to \'{}\'...'.format(args.output))

if __name__ == '__main__':
    main()
