import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence(=2d point), build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  #12 here is from P11 to P34
  constraint_matrix = np.zeros((num_corrs * 2, 12))
  zero = np.zeros(4)
  for i in range(num_corrs):
    X = np.array([*points3D[i], 1])
    x = points2D[i]
    # first row = [0t -Xt x2Xt]
    constraint_matrix[2*i] = np.concatenate([zero,-X, x[1] * X])
    # second row = [Xt 0t -x1Xt]
    constraint_matrix[2*i+1] = np.concatenate([X, zero, -x[0] * X])

  return constraint_matrix