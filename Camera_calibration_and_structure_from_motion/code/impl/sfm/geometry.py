import numpy as np
from numpy.core.numeric import full

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # Normalize coordinates (to points on the normalized image plane)
  K_inv = np.linalg.inv(K)
  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  normalized_kps1 = (K_inv @ np.c_[im1.kps,np.ones(len(im1.kps))].T).T
  normalized_kps2 = (K_inv @ np.c_[im2.kps,np.ones(len(im2.kps))].T).T

  # Assemble constraint matrix
  constraint_matrix = np.zeros((matches.shape[0], 9)) # 5765x9
  for i in range(matches.shape[0]):
    x1 = normalized_kps1[matches[i,0]]
    x2 = normalized_kps2[matches[i,1]]
    # Add the constraints
    constraint_matrix[i] = np.concatenate([x1[0]*x2, x1[1]*x2, x1[2]*x2])
  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape((3,3))
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  P,D,Q = np.linalg.svd(E_hat)
  D = np.eye(3)
  D[-1,-1] = 0
  E = P@D@Q
  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)
  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), axis=1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), axis=1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], axis=0)

  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  # For each match where the point is in both images, compute it's position
  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]
  
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
  # Z axis is index 2
  to_add = np.zeros((1,4))
  to_add[-1,-1] = 1
  T1= np.append((np.append(R1, np.expand_dims(t1, 1), axis=1)),to_add,axis=0)
  T2= np.append((np.append(R2, np.expand_dims(t2, 1), axis=1)),to_add,axis=0)


  #im_1_proj = (P1 @ MakeHomogeneous(points3D.T))
  #im_2_proj = (P2 @ MakeHomogeneous(points3D.T))
  im_1_vision = (T1 @ MakeHomogeneous(points3D.T))
  im_2_vision = (T2 @ MakeHomogeneous(points3D.T))
  f = np.logical_and(im_1_vision[2,:] >= 0, im_2_vision[2,:] >= 0)
  points3D = points3D[f]
  im1_corrs = im1_corrs[f]
  im2_corrs = im2_corrs[f]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  normalized_points2D = (np.linalg.inv(K) @ np.c_[points2D,np.ones(len(points2D))].T).T

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):
  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  e_img = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}
  img_all_corrs = list()
  for reg_img_name in registered_images:
    e_reg_img = images[reg_img_name]
    e_matches = GetPairMatches(image_name , reg_img_name, matches)
    points3D_to_add, img_corrs, reg_img_corrs = TriangulatePoints(K,e_img,e_reg_img,e_matches)

    # First item of the tuple is the offset and the second is the list of indices for the 2d
    img_all_corrs.append(tuple([len(points3D),list(img_corrs)]))
    corrs[reg_img_name] = [tuple([len(points3D), list(reg_img_corrs)])]

    points3D = np.concatenate((points3D, points3D_to_add), axis=0)
    
  corrs[image_name] = img_all_corrs
  
  return points3D, corrs
  
