# CSC320 Winter 2018
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    D_shape = np.shape(source_patches)
    #check if best_D exist
    if best_D is None:
        best_D=np.full((D_shape[0], D_shape[1]), np.inf)


    if odd_iteration:
        for i in range(D_shape[0]):
            for j in range(D_shape[1]):
                if not propagation_enabled:
                    self_diff = np.reshape(source_patches[i, j], -1) - np.reshape(
                        target_patches[new_f[i, j, 0], new_f[i, j, 1]], -1)
                    self_diff = np.abs(np.where(np.isnan(self_diff), 0, self_diff))
                    # self_D = np.sqrt(self_diff.dot(self_diff))
                    self_D = np.linalg.norm(self_diff)
                    if self_D < best_D[i, j]:
                        best_D[i, j] = self_D
                        # D_value of left
                    if i - 1 >= 0 and 0 <= i + new_f[i - 1, j, 0] <= (D_shape[0] - 1) and 0 <= j + new_f[i - 1, j, 1] <= (D_shape[1] - 1):
                        left_diff = np.reshape(source_patches[i, j], -1) - np.reshape(
                            target_patches[i + new_f[i - 1, j, 0], j + new_f[i - 1, j, 1]], -1)
                        left_diff = np.abs(np.where(np.isnan(left_diff), 0, left_diff))
                        # left_D = np.sqrt(left_diff.dot(left_diff))
                        left_D = np.linalg.norm(left_diff)
                        if left_D < best_D[i, j]:
                            best_D[i, j] = left_D
                            new_f[i, j] = new_f[i - 1, j]
                    # D_value of down
                    if j - 1 >= 0 and 0 <= i + new_f[i, j - 1, 0] <= (D_shape[0] - 1) and 0 <= j + new_f[
                        i, j - 1, 1] <= (D_shape[1] - 1):
                        down_diff = np.reshape(source_patches[i, j], -1) - np.reshape(target_patches[i + new_f[i, j - 1, 0], j + new_f[i, j - 1, 1]], -1)
                        non_nan = np.count_nonzero(~np.isnan(down_diff))
                        down_diff = np.abs(np.where(np.isnan(down_diff), 0, down_diff))
                        down_D = np.linalg.norm(down_diff)
                        if down_D < best_D[i, j]:
                            best_D[i, j] = down_D
                            new_f[i, j] = new_f[i, j - 1]

                if not random_enabled:
                    temp_w = w
                    exp = 1
                    while alpha * temp_w >= 1:
                        R = np.array([(alpha * temp_w * (np.random.uniform(-1, 1, 1))[0]).astype(int),
                                      (alpha * temp_w * (np.random.uniform(-1, 1, 1))[0]).astype(int)])
                        u0 = np.array([i + new_f[i, j, 0] + R[0], j + new_f[i, j, 1] + R[1]])
                        new_i, new_j = u0[0].astype(int), u0[1].astype(int)
                        if 0 <= new_i <= (D_shape[0] - 1) and 0 <= new_j <= (D_shape[1] - 1):
                            s_patch, t_patch = source_patches[i, j], target_patches[new_i, new_j]
                            # calculate patch D-value
                            diff = np.reshape(s_patch, -1) - np.reshape(t_patch, -1)
                            diff = np.abs(np.where(np.isnan(diff), 0, diff))
                            # new_D = np.sqrt(diff.dot(diff))
                            new_D = np.linalg.norm(diff)
                            if new_D < best_D[i, j]:
                                best_D[i, j] = new_D
                                new_f[i, j] = np.array([new_i - i, new_j - j])
                        temp_w = (alpha ** exp) * temp_w
                        exp += 1

    else:
        for i in range(D_shape[0]-1, -1, -1):
            for j in range(D_shape[1]-1, -1 , -1):
                if not propagation_enabled:
                    self_diff = np.reshape(source_patches[i, j], -1) - np.reshape(target_patches[new_f[i, j, 0], new_f[i, j, 1]], -1)
                    self_diff = np.abs(np.where(np.isnan(self_diff), 0, self_diff))
                    self_D = np.linalg.norm(self_diff)
                    if self_D < best_D[i, j]:
                        best_D[i, j] = self_D

                    # D_value of right
                    if i + 1 <= (D_shape[0] - 1) and 0 <= i + new_f[i + 1, j, 0] <= (D_shape[0] - 1) and 0 <= j + \
                            new_f[i + 1, j, 1] <= (D_shape[1] - 1):
                        right_diff = np.reshape(source_patches[i, j], -1) - np.reshape(
                            target_patches[i + new_f[i + 1, j, 0], j + new_f[i + 1, j, 1]], -1)
                        right_diff = np.abs(np.where(np.isnan(right_diff), 0, right_diff))
                        right_D = np.linalg.norm(right_diff)
                        if right_D < best_D[i, j]:
                            best_D[i, j] = right_D
                            new_f[i, j] = new_f[i + 1, j]
                    # D_value of up
                    if j + 1 <= (D_shape[1] - 1) and 0 <= i + new_f[i, j + 1, 0] <= (D_shape[0] - 1) and 0 <= j + \
                            new_f[i, j + 1, 1] <= (D_shape[1] - 1):
                        up_diff = np.reshape(source_patches[i, j], -1) - np.reshape(target_patches[i + new_f[i, j + 1, 0], j + new_f[i, j + 1, 1]], -1)
                        up_diff = np.abs(np.where(np.isnan(up_diff), 0, up_diff))
                        up_D = np.linalg.norm(up_diff)
                        if up_D < best_D[i, j]:
                            best_D[i, j] = up_D
                            new_f[i, j] = new_f[i, j + 1]

                if not random_enabled:
                    temp_w = w
                    exp = 1
                    while alpha * temp_w >= 1:
                        R = np.array([(alpha * temp_w * (np.random.uniform(-1, 1, 1))[0]).astype(int),
                                      (alpha * temp_w * (np.random.uniform(-1, 1, 1))[0]).astype(int)])
                        u0 = np.array([i + new_f[i, j, 0] + R[0], j + new_f[i, j, 1] + R[1]])
                        new_i, new_j = u0[0].astype(int), u0[1].astype(int)
                        if 0 <= new_i <= (D_shape[0] - 1) and 0 <= new_j <= (D_shape[1] - 1):
                            s_patch, t_patch = source_patches[i, j], target_patches[new_i, new_j]
                            # calculate patch D-value
                            diff = np.reshape(s_patch, -1) - np.reshape(t_patch, -1)
                            diff = np.abs(np.where(np.isnan(diff), 0, diff))
                            new_D = np.linalg.norm(diff)
                            if new_D < best_D[i, j]:
                                best_D[i, j] = new_D
                                new_f[i, j] = np.array([new_i - i, new_j - j])
                        temp_w = (alpha ** exp) * temp_w
                        exp += 1
    #############################################

    return new_f, best_D, global_vars


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    combine_dis = make_coordinates_matrix(target.shape) + f
    x,y = combine_dis[:,:,0], combine_dis[:,:,1]
    rec_source = target[x,y,:]
    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
