from typing import Tuple
import numpy as np
from math import sin, cos
from pyquaternion import Quaternion as q
from shapely.geometry import box
import cv2


def get_rot_bounding_box_experimental(img, box_points, out_size):

    # the order of the box points: bottom left, top left, top right,
    # bottom right

    # get width and height of the detected rectangle
    height = int(out_size[0])
    width = int(out_size[1])
    src_pts = np.column_stack((box_points[:4, 1], box_points[:4, 0])).astype(np.float32)
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_box = box(0, 0, height - 1, width - 1)
    dst_pts = np.array(dst_box.exterior.coords)
    dst_pts = np.column_stack((dst_pts[:4, 1], dst_pts[:4, 0])).astype(np.float32)

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # M = cv2.getAffineTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    # warped = cv2.warpAffine(img, M, (height, width))
    return warped


def get_rot_bounding_box(
    image, box: Tuple[int, int, int, int], angle, *, scale: float = 1
):
    """
    return a rotated image box
    -----
    Args:
    box: Tuple[center_x, center_y, height, width]
    rotation in degrees

    Returns:
    The cropped and rotated patch
    """
    raise DeprecationWarning

    rotation_matrix = cv2.getRotationMatrix2D((box[1], box[0]), angle, scale)

    rcenter = np.array([box[0], box[1], 1])

    o = np.dot(rotation_matrix, rcenter)

    rotation_matrix[0, 2] -= o[0]
    # rotation_matrix[1, 2] -= o[1]

    patch = cv2.warpAffine(
        image, rotation_matrix, (box[3], box[2]), flags=cv2.WARP_INVERSE_MAP
    )

    return patch


def road_boundary_direction_map(road_boundary_image):

    transform_map = cv2.distanceTransform(
        road_boundary_image,
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
    )

    deriv_x = cv2.Sobel(transform_map, cv2.CV_32F, dx=1, dy=0)
    deriv_y = cv2.Sobel(transform_map, cv2.CV_32F, dx=0, dy=1)

    # now normalize
    deriv = np.stack((deriv_x, deriv_y), axis=-1)

    magnitude = np.linalg.norm(deriv, axis=2)
    magnitude[magnitude == 0] = 1.0
    normalized_vector_field = deriv / magnitude[:, :, np.newaxis]

    angle_field = np.arctan2(
        normalized_vector_field[:, :, 0], normalized_vector_field[:, :, 1]
    )

    return normalized_vector_field, angle_field


def inverse_distance_map(
    road_boundary_image,
    truncation_thld: float = 5.0,
    map_resolution: float = 0.1,
):

    transform_map = cv2.distanceTransform(
        road_boundary_image,
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
    )
    # truncate
    truncation_value = truncation_thld / map_resolution
    truncate_map = np.minimum(transform_map, truncation_value)
    # inverse
    inverse_distance_map = truncation_value - truncate_map

    return inverse_distance_map


def end_point_heat_map(ground_truth_img):
    # find endpoints
    end_point_map = np.zeros_like(ground_truth_img).astype(np.float32)
    mask = ground_truth_img == 0
    np.add(
        end_point_map[:1, :, :],
        1,
        out=end_point_map[:1, :, :],
        where=mask[:1, :, :],
    ),
    np.add(
        end_point_map[:, :1, :],
        1,
        out=end_point_map[:, :1, :],
        where=mask[:, :1, :],
    ),
    np.add(
        end_point_map[-1:, :, :],
        1,
        out=end_point_map[-1:, :, :],
        where=mask[-1:, :, :],
    ),
    np.add(
        end_point_map[:, -1:, :],
        1,
        out=end_point_map[:, -1:, :],
        where=mask[:, -1:, :],
    ),

    end_point_map = cv2.GaussianBlur(end_point_map, (55, 55), 8)
    # normalize in interval [0, 1)
    np.divide(end_point_map, np.max(end_point_map) + 1e-12, out=end_point_map)

    # create a gaussian kernel
    return end_point_map


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import pickle

    with open(
        "/home/dominic/data/nuscenes_train_big_ro_big_roii/data/scene_6_sample_39_data.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)

    ground_truth_img = data["ground_truth"]
    ep = end_point_heat_map(ground_truth_img)

    heat_end = (ep * 255).astype(np.uint8)
    heat_end = cv2.applyColorMap(heat_end, cv2.COLORMAP_JET)
    cv2.imwrite("/home/dominic/data/point_heat_map.png", heat_end)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(np.squeeze(ground_truth_img))
    ax[1].imshow(np.squeeze(ep))
    plt.show()
    """
    image = cv2.imread("/home/dominic/repos/lyft_dataset_tools/data/label.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = 255 - image

    inverse_distance_map = inverse_distance_map(image)
    boundary_direction, angle_map = road_boundary_direction_map(image)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(image, cmap="gray", vmin=0, vmax=255)
    ax[0][1].imshow(inverse_distance_map)
    ax[1][0].imshow(angle_map)
    plt.show()
    """
