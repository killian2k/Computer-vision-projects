import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = np.zeros((nPointsX*nPointsY,2))  # numpy array, [nPointsX*nPointsY, 2]

    H, W = img.shape
    Ys = np.floor(np.linspace(border, H - border, nPointsY, endpoint=False))
    Xs = np.floor(np.linspace(border, W - border, nPointsX, endpoint=False))
    xv, yv = np.meshgrid(Xs, Ys, indexing='ij')
    vPoints[:, 0] = xv.flatten()
    vPoints[:, 1] = yv.flatten()

    return vPoints.astype(np.int32)


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    '''
    Next, we describe each feature (grid point) by a local descriptor.
    We will use the well known histogram of oriented gradients (HOG) descriptor.
    Implement the function descriptors hog which should compute
    one 128-dimensional descriptor for each of the N 2-dimensional grid points (contained in the N × 2
    input argument vPoints).
    The descriptor is defined over a 4 × 4 set of cells placed around the grid point i (see Fig. 1).

    For each cell (containing cellWidth × cellHeight pixels, which we will choose to be cellWidth = cellHeight = 4),
    create an 8-bin histogram over the orientation of the gradient at each pixel within the cell. 16 cells * 8 = 128
    Then concatenate the 16 resulting histograms (one for each cell) to produce a 128 dimensional vector for every
    grid point. Finally, the function should return a N × 128 dimensional feature matrix.
    '''
    global grad_magn
    nBins = 8
    number_of_cell = 4
    angle = 360
    step_size = angle / nBins
    pix_per_batch_y = cellHeight * number_of_cell
    pix_per_batch_x = cellWidth * number_of_cell
    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)
    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        x, y = tuple(vPoints[i])
        local_grad_x = grad_x[y:y+pix_per_batch_y, x:x+pix_per_batch_x].astype(np.float64)
        local_grad_y = grad_y[y:y+pix_per_batch_y, x:x+pix_per_batch_x].astype(np.float64)
        grad_magn = np.sqrt(local_grad_x**2 + local_grad_y**2)
        grad_angle = np.rad2deg(np.arctan2(local_grad_y, local_grad_x)) % angle
        #print(grad_magn.shape, grad_angle.shape)
        #histogram, _ = np.histogram(grad_angle, bins=nBins, weights=grad_magn)
        grad_bin = grad_angle // step_size
        histogram = np.zeros((*grad_magn.shape, 8))
        for m in range(nBins):
            histogram[:, :, m] = np.where(grad_bin == m, grad_magn, 0)
        histogram = histogram.reshape((number_of_cell, cellHeight, number_of_cell, cellWidth, 8)).sum((3, 1))
        descriptors.append(histogram.flatten())

    descriptors = np.asarray(
        descriptors)  # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # todo
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        vFeatures.append(descriptors_hog(img, vPoints, cellWidth, cellHeight))

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]

    # Cluster the features using K-Means
    print('clustering ...', vFeatures.shape)
    km = KMeans(n_clusters=k, max_iter=numiter)
    kmeans_res = km.fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    histo = np.zeros(len(vCenters))

    idx, _ = findnn(vFeatures, vCenters)
    idx = np.array(idx)
    #print("Dist: ", dist)

    #for i in range(vFeatures.shape[0]):
    #    featureVector = vFeatures[i] # vector size D
    #    index = np.argmin(((vCenters-featureVector)**2).sum(1))
    #    histo[index]+=1
    #for i in range(len(histo)):
    histo = [np.count_nonzero(idx == i) for i in range(len(vCenters))]
    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        vPoints = grid_points(img, nPointsX, nPointsY, border)
        vFeatures = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        bowHist = bow_histogram(vFeatures, vCenters)
        vBoW.append(bowHist)

    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW


def bow_recognition_nearest(histogram, vBoWPos, vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = findnn(histogram,vBoWPos)[1], findnn(histogram, vBoWNeg)[1]#((vBoWPos-histogram)**2).sum(1).min(), ((vBoWNeg-histogram)**2).sum(1).min()

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor

    return 1 if DistPos < DistNeg else 0


if __name__ == '__main__':
    np.random.seed(30)
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    k = 220#Best: 300#200
    numiter = 10000#Best: 5000

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
