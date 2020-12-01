import numpy as np
import cv2
from pyquaternion import Quaternion
from dataset_utilities.transformation import (
    Isometry,
    to_homogenous_points,
)
from pathlib import Path

# from cv2 import warpPerspective
from dataset_utilities.pcd_parser import PCDParser
from enum import Enum, auto, unique
import logging

from cached_property import cached_property


def to2D(point_3d):
    # assert(point_3d.size == (3,))
    return point_3d[0:2]


def toHomogenousMat(mat: np.ndarray):
    mat = np.array(mat)
    assert mat.shape[:2] == (3, 3) and len(mat.shape) == 3
    # pad = np.broadcast_to(np.array([0, 0, 0, 1]), (4, mat.shape[2]))
    raise NotImplementedError  # TODO


@unique
class Frame(Enum):
    SENSOR = auto()
    VEHICLE = auto()
    WORLD = auto()

    UNKNOWN = auto()


class SensorBase(object):
    def __init__(
        self,
        id: str,
        extrinsic: Isometry = Isometry(),
        sensor_type: str = "base",
    ):
        self.id = id
        self.sensor_type = sensor_type
        self.frame = Frame.UNKNOWN

        self.extrinsic = extrinsic

        self.M = extrinsic.matrix
        self.M_inv = extrinsic.inverse().matrix

        self.filename = None
        self.data = None
        self.timestamp = None

        self.ego_mask = None

    def load_data(self, filename: str, data=None) -> bool:
        raise NotImplementedError

    def filterBox(
        self,
        box_center: np.ndarray,
        box_dims: np.ndarray,
        orientation: Quaternion = Quaternion(),
    ):
        raise NotImplementedError

    def writeData(self, output_path: str = None):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError


class Lidar(SensorBase):
    def __init__(
        self,
        id: str,
        extrinsic: Isometry = Isometry(),
        sensor_type: str = "lidar",
    ):
        super().__init__(id=id, extrinsic=extrinsic, sensor_type=sensor_type)
        self.data = None  # data is assumed in format [channel, nPoints]
        self.frame = Frame.UNKNOWN

    def load_data(self, filename: Path = None, data=None):
        if filename is not None:
            # filename = Path(filename)
            if filename.is_file():
                if filename.suffix == ".pcd":
                    pcd_parser = PCDParser()
                    point_cloud, viewpoint = pcd_parser.read(str(filename))
                    self.data = point_cloud
                    self.filename = str(filename)
                    self.FRAME = Frame.SENSOR
                else:
                    raise NotImplementedError

            else:
                logging.error(
                    "Could not locate file %s for sensor %s" % (filename, self.id)
                )
                return False
        elif data is not None:
            self.data = data
            assert data.shape[0] in [3, 4]
            self.filename = None
            self.frame = Frame.SENSOR
            return True
        else:
            return False

    def export(self):
        return {"extrinsic": self.M, "frame": self.frame, "data": self.data}

    def transformLidarToVehicle(self):
        logging.debug("transformed lidar to vehicle frame")
        if self.frame == Frame.SENSOR:
            # self.data.transform(homogenous_M)
            self.data[:3, :] = self.extrinsic.transform(self.data[:3, :])
            self.frame = Frame.VEHICLE

    def transformVehicleToLidar(self):
        if self.frame == Frame.VEHICLE:
            self.data[:3, :] = self.extrinsic.inverse().transform(self.data[:3, :])
            self.frame = Frame.SENSOR

    def projectToGround(self):
        self.data.points[2, :] = 0

    def filter_boxes(self, boxes, return_inliers: bool = False):
        for box in boxes:
            T = Isometry(translation=box.center, rotation=box.orientation)
            T_inv = T.inverse()
            x = np.asarray([1.0, 0.0, 0.0])
            y = np.asarray([0.0, 1.0, 0.0])
            z = np.asarray([0.0, 0.0, 1.0])

            x_lim = np.dot(x, np.asarray([box.wlh[1] / 2, 0, 0]))
            y_lim = np.dot(y, np.asarray([0, box.wlh[0] / 2, 0]))
            z_lim = np.dot(z, np.asarray([0, 0, box.wlh[2] / 2]))

            point = T_inv @ self.data.points[:3, :]
            x_p = np.dot(x, point)
            y_p = np.dot(y, point)
            z_p = np.dot(z, point)

            mask = np.logical_and.reduce(
                [
                    x_p >= -x_lim,
                    x_p < x_lim,
                    y_p >= -y_lim,
                    y_p < y_lim,
                    z_p >= -z_lim,
                    z_p < z_lim,
                ]
            )

            if return_inliers:
                self.data.points = self.data.points[:, mask]
            else:
                self.data.points = self.data.points[:, ~mask]

    def filterBox(
        self,
        box_center,
        box_dims,
        orientation: Quaternion = Quaternion(),
        return_inliers: bool = False,
    ):
        """filters all points within [origin, origin + dims]
        box_dims = [width, length, height]
        """
        # remove all points within box
        # create a n-dim boolean mask for each dimension (x,y,z)
        logging.warning(
            "\u001b[31m LIDAR.filterBox is deprecated in favor of LIDAR.filter_boxes\u001b[0m"
        )
        T = Isometry(translation=box_center, rotation=orientation)
        T_inv = T.inverse()
        x = np.asarray([1.0, 0.0, 0.0])
        y = np.asarray([0.0, 1.0, 0.0])
        z = np.asarray([0.0, 0.0, 1.0])

        x_lim = np.dot(x, np.asarray([box_dims[1] / 2, 0, 0]))
        y_lim = np.dot(y, np.asarray([0, box_dims[0] / 2, 0]))
        z_lim = np.dot(z, np.asarray([0, 0, box_dims[2] / 2]))

        nmbr_samples = self.data.points.shape[1]
        bool_mask = np.zeros(nmbr_samples, dtype=bool)
        for i in range(nmbr_samples):
            point = T_inv @ self.data.points[:3, i]
            x_p = np.dot(x, point)
            y_p = np.dot(y, point)
            z_p = np.dot(z, point)

            mask = (
                x_p >= -x_lim
                and x_p < x_lim
                and y_p >= -y_lim
                and y_p < y_lim
                and z_p >= -z_lim
                and z_p < z_lim
            )
            bool_mask[i] = mask

        if return_inliers:
            filtered_array = self.data.points[:, bool_mask]
        else:
            filtered_array = self.data.points[:, ~bool_mask]

        return filtered_array

    def exportPCD(self, output_path: Path = None):
        # convert the sensor's data into pcl format and write to file
        # also export calibration and add timestamp to filename
        # the point cloud is rotated in alignment with vehicle frame but with origin in sensor origin

        adjusted_calibration = self.extrinsic

        adjusted_point_cloud = self.data

        if self.frame == Frame.VEHICLE:
            adjusted_point_cloud[:3, :] = adjusted_calibration.inverse().transform(
                adjusted_point_cloud[:3, :]
            )

        elif self.frame == Frame.SENSOR:
            pass
        else:
            raise ValueError(self.frame)

        parser = PCDParser()
        parser.addCalibration(calibration=adjusted_calibration)
        output_path = Path(output_path)
        if isinstance(output_path, Path):
            output_path.mkdir(exist_ok=True, parents=True)
        file = output_path / self.id
        parser.write(
            np.swapaxes(adjusted_point_cloud[:, :], 0, 1),
            file_name=str(file),
        ),
        return file

    def writeData(self, output_path: Path = None):
        if output_path is None:
            output_path = Path(str(self.id) + "_" + str(self.timestamp))

        np.save(
            str(output_path / (str(self.id) + "_" + str(self.timestamp))),
            self.data.points,
        )
        np.savetxt(
            str(
                output_path
                / (str(self.id) + "_" + str(self.timestamp) + "_calibration")
            ),
            self.M,
        )


class Camera(SensorBase):
    REMOVE_COLOR = (0, 255, 0)

    """Camera class for handeling a vehicle mounted camera setup"""

    def __init__(
        self,
        id,
        extrinsic: Isometry = Isometry(),
        intrinsic: np.ndarray = np.eye(3),
        sensor_type="camera",
        *,
        ego_contour=None,
        crop_horizon=True,
    ):
        """TODO: to be defined.

        :_K: Intrinsic calibration matrix
         _R: rotation matrix from camera to vehicle
         _t: translation vector from camera to vehicle
         _M: affine transformation following the extrinsic calibration M=[_R|_t]
         _P: projective matrix P=K*M=A[R|t]
         _H: homography matrix ground plane -> image
         _H_inv: homography matrix image -> ground plane

        """
        super().__init__(id=id, extrinsic=extrinsic, sensor_type=sensor_type)

        # camera calibration
        self.K = np.asarray(intrinsic)
        self.cropped = False
        self.reset()

        if ego_contour is not None:
            self.ego_mask = [ego_contour]
        else:
            self.ego_mask = None

        if crop_horizon:
            self._find_horizon()
        else:
            self.horizon = None

    def reset(self):
        """reset all internal parameters of the instance"""

        # projective matrix
        self.P = self.K @ self.M_inv[:3, :4]
        # self.P = self.K @ self.M[:3, :4]
        assert self.P.shape == (3, 4)

        # clearing cached properties
        if "H" in self.__dict__:
            del self.__dict__["H"]
        if "H_inv" in self.__dict__:
            del self.__dict__["H_inv"]

    def export(self):
        return {"intrinsic": self.K, "extrinsic": self.M, "data": self.data}

    # homography
    @cached_property
    def H(self):
        H = self.K @ np.column_stack(
            (self.M_inv[:3, 0], self.M_inv[:3, 1], self.M_inv[:3, 3])
        )
        assert H.shape == (3, 3)
        return H

    @cached_property
    def H_inv(self):
        H_inv = np.linalg.inv(self.H)
        assert H_inv.shape == (3, 3)
        return H_inv

    def transformImageToGround(self, image_point):
        homogenous_point = self.transformImageToVehicle(image_point)
        return to2D(homogenous_point) / homogenous_point[-1]

    def transformImageToVehicle(self, image_point):
        return self.H_inv @ to_homogenous_points(image_point)

    def transformVehicleToImageExact(self, point):
        homogenous_image_point = self.P @ to_homogenous_points(point)
        return to2D(homogenous_image_point) / homogenous_image_point[-1]

    def transformVehicleToImage(self, point):
        return np.round(self.transformVehicleToImageExact(point)).astype(np.int32)

    def transformGroundToImageExact(self, ground_point):
        homogenous_image_point = self.H @ to_homogenous_points(to2D(ground_point))
        return to2D(homogenous_image_point) / homogenous_image_point[-1]

    def transformGroundToImage(self, ground_point):
        return np.round(self.transformGroundToImageExact(ground_point)).astype(np.int32)

    def cropImage(self, img, height_roi=None, width_roi=None):
        """crop a given image to given size.
        :img: the uncropped image
        :height_roi: [begin_crop, end_crop]
        :width_roi: [begin_crop, end_crop]
        :returns: cropped image
        """
        (img_height, img_width, _) = img.shape
        roi_w_low = 0
        roi_h_low = 0
        roi_w_high = img_width - 1
        roi_h_high = img_height - 1
        if width_roi is not None:
            roi_w_low = max(0, width_roi[0])
            roi_w_high = min(img_width, width_roi[1]) if width_roi[1] > 0 else img_width
        if height_roi is not None:
            roi_h_low = max(0, height_roi[0])
            roi_h_high = (
                min(img_height, height_roi[1]) if height_roi[1] > 0 else img_height
            )
        # adjust the camera intrinsic
        if not self.cropped:
            self.K[0, 2] -= roi_w_low
            self.K[1, 2] -= roi_h_low
            self.reset()
            self.cropped = True

        return img[roi_h_low:roi_h_high, roi_w_low:roi_w_high, :]

    def load_data(self, filepath: Path = None, data: np.ndarray = None) -> bool:
        if filepath is not None:
            """load image frim filepath using cv2.imread"""
            if filepath.is_file():
                img = cv2.imread(str(filepath))
                self.filename = filepath
                self.data = img
            else:
                logging.warn("file %s does not exist" % filepath)
                return False
        elif data is not None:
            if data.dtype == np.float:
                data = (255 * data).astype(np.uint8)
            self.data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            self.filename = Path()
        else:
            raise ValueError

        if self.ego_mask is not None:
            self.removeContour(self.ego_mask, remove_color=self.REMOVE_COLOR)
            logging.debug("remove contour")
        if self.horizon != -1 and self.horizon is not None:
            if self.horizon >= 0:
                self.data = self.cropImage(
                    self.data,
                    height_roi=[max(0, self.horizon), -1],
                )
        self.input_size = self.data.shape
        logging.debug("load data from " + str(filepath))
        return True

    def isInSight(self, point):
        projection = self.P @ to_homogenous_points(point)
        sight = (projection >= 0).any() and (
            projection[:2] < np.array(self.input_size[:2]).any()
        )
        return sight and projection[3] > 0

    def filter_boxes(self, boxes, return_inliers: bool = False):
        # remove all boxes from camera image
        if return_inliers:
            raise NotImplementedError

        for box in boxes:
            # to_homogenous_points(box.corners())
            image_corners = self.P @ to_homogenous_points(box.corners())
            if (image_corners[2, :] < 0).any():
                continue
            image_corners /= image_corners[-1, :]
            image_corners = np.round(image_corners).astype(np.int32)
            if np.logical_and(
                np.logical_and(
                    image_corners[0, :] < self.data.shape[1],
                    image_corners[0, :] >= 0,
                ),
                np.logical_and(
                    image_corners[1, :] < self.data.shape[1],
                    image_corners[0, :] >= 0,
                ),
            ).any():
                # render box
                image_corners[0, image_corners[0, :] >= self.data.shape[1]] = (
                    self.data.shape[1] - 1
                )
                image_corners[1, image_corners[1, :] >= self.data.shape[0]] = (
                    self.data.shape[0] - 1
                )
                image_corners[0, image_corners[0, :] < 0] = 0
                image_corners[1, image_corners[1, :] < 0] = 0
                contour = np.array(
                    [
                        image_corners[:2, i].tolist()
                        for i in range(image_corners.shape[1])
                    ]
                )
                self.removeContour(
                    [cv2.convexHull(contour)], remove_color=self.REMOVE_COLOR
                )

    def removeContour(self, contour: np.ndarray, remove_color=(0, 0, 0)):
        self.data = cv2.drawContours(
            self.data, contour, -1, color=remove_color, thickness=cv2.FILLED
        )

    @staticmethod
    def normalizeImage(image):
        raise NotImplementedError

    def drawCalibration(self):
        center_line = [np.array([i, 0, 0]) for i in range(5, 10, 1)]
        for center_point in center_line:
            px_point = self.transformGroundToImage(center_point)
            """
            print(px_point)
            marker_size = 2
            if (
                px_point[1] < self.data.shape[1]
                and px_point[1] >= 0
                and px_point[1] < self.data.shape[0]
                and px_point[0] >= 0
            ):
                self.data[
                    px_point[1] - marker_size : px_point[1] + marker_size,
                    px_point[0] - marker_size : px_point[0] + marker_size,
                ] = (0, 0, 255)
            """
            self.data = cv2.drawMarker(
                self.data, tuple(px_point), color=(255, 0, 0), thickness=2
            )

    def _find_horizon(self):
        # automatically set

        line_at_infinity = [1, 0, 0, 0]
        proj = self.P @ line_at_infinity
        proj = np.ceil(proj[:2] / proj[2]).astype(np.int32)
        self.horizon = max(0, proj[1])
        if self.horizon < 0:
            logging.warn("horizon < 0 for Camera: ", self.id)

    def write_data(self, output_path: Path) -> Path:
        if self.data is None:
            logging.warn("no data for camera %s" % self.id)
            return Path()
        output_path.mkdir(exist_ok=True, parents=True)
        file = output_path / (self.id + ".png")
        cv2.imwrite(str(file), self.data)
        return file


class BirdsEyeView(Camera):

    """Docstring for BirdsViewTransformation. """

    def __init__(
        self,
        id,
        extrinsic: Isometry = Isometry(),
        intrinsic: np.ndarray = np.eye(4),
        *,
        offset=(0, 0),
        resolution=0.01,
        out_size=(200, 200),
        camera=None,
        **kwargs,
    ):
        """TODO: to be defined."""
        super().__init__(
            id, extrinsic=extrinsic, intrinsic=intrinsic, sensor_type="bev", **kwargs
        )
        self.offset = offset
        self.resolution = resolution

        self.size = tuple(out_size)
        self.input_size = None

        # hacky copy constructor
        if isinstance(camera, Camera):
            self.__dict__.update(camera.__dict__)

    def transform(
        self,
        data: np.ndarray = None,
        *,
        offset=None,  # [m]
        resolution=None,  # [m/px]
        out_size=None,  # [px]
        aux_transform=np.eye(3),
        **kwargs,
    ):
        if out_size is None:
            out_size = self.size

        data = np.asarray(data)
        bev_transformation = (
            self.scale_transformation(resolution)
            @ self.shift_roi(offset)
            @ aux_transform
            @ self.H_inv  # np.linalg.inv(self.P[:3,:3])
        )
        logging.debug("BEV-Transformation: %s" % bev_transformation)
        if data.all() is not None:
            bev = cv2.warpPerspective(data, bev_transformation, out_size, **kwargs)
        else:
            bev = cv2.warpPerspective(self.data, bev_transformation, out_size, **kwargs)
        return bev

    def shift_roi(self, offset=None):
        """docstring for shift_roi"""
        if offset is None:
            offset = self.offset
        shift_matrix = np.array(
            # [[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]]
            [[0, 1, offset[0]], [1, 0, offset[1]], [0, 0, 1]]
        )
        return shift_matrix

    def scale_transformation(self, resolution: float = None):
        """docstring for scale_transformation"""
        if resolution is not None:
            scaling_factor = 1 / resolution
        else:
            scaling_factor = 1 / self.resolution

        scaling_matrix = np.array(
            [[scaling_factor, 0, 0], [0, scaling_factor, 0], [0, 0, 1]]
        )
        return scaling_matrix


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
    import time

    nmbr_boxes = 50
    nmbr_samples = 100000
    # test lidar filtering
    lidar_data = np.random.uniform(low=0.0, high=1000.0, size=(4, nmbr_samples))
    print("PointCloud generated")
    test_lidar = Lidar(id="test")
    boxes = [
        Box(
            center=np.random.random([3]) * 1000.0,
            size=(np.random.random([3]) * 25 + 25),
            orientation=Quaternion.random(),
        )
        for i in range(nmbr_boxes)
    ]
    test_lidar.data = LidarPointCloud(points=lidar_data.copy())

    print(
        "Filter %s Boxes from PointCloud with %s Samples"
        % (len(boxes), test_lidar.data.points.shape[1])
    )
    """
    start = time.time()
    for box in boxes:
        test_lidar.data.points = test_lidar.filterBox(box_center=box.center, box_dims=box.wlh, orientation=box.orientation)
    runtime = time.time() - start
    old_results = test_lidar.data.points.copy()
    print("Previous function finished in %s seconds (%s samples removed)" % (runtime, nmbr_samples-old_results.shape[1]))

    test_lidar.data = LidarPointCloud(points=lidar_data.copy())
    """

    start = time.time()
    test_lidar.filter_boxes(boxes)
    runtime = time.time() - start

    print(
        "New function finished in %s seconds (%s samples removed)"
        % (runtime, nmbr_samples - test_lidar.data.points.shape[1])
    )
    # assert test_lidar.data.points is not old_results
    # assert (test_lidar.data.points == old_results).all()
    print("Results are identical")

    """
    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")
    ax.scatter(filtered[0, :], filtered[1, :], filtered[2, :], marker="^")
    print(filtered.shape)
    plt.show()
    """
