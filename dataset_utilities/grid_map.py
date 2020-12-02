"""
This class aim at the creation of a dense elevation grid from several lidar sweeps over time

Based on a given vehicle pose a certain roi is computed as grid updated until leaving the roi

For Lidar Measurements:
    1. for every point create a ray
    2. compute the height and occupations status of the ray for every cell it passes
    3. refine the grid by closing holes
"""
import logging
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

# from image_utils import get_rot_bounding_box
from dataset_utilities.pcd_parser import PCDParser
from dataset_utilities.camera import BirdsEyeView
from dataset_utilities.transformation import Isometry, to_homogenous_points
#from bev_stitcher import Stitcher


class RayArray:
    """
    Class casting rays for every point in a point cloud
    ------
    Args:
    origin: sensor origin. the values are supposed to be discretized.
    points: [3,n] numpy array. the first two dimensions are supposed to be dicretized.

    """

    def __init__(self, origin, points):

        if origin.ndim == 1:
            origin = np.expand_dims(origin, 1)

        points = points[:, ~(points[:2, :] == origin[:2, :]).all(axis=0)]

        # cast x,y to integer
        self.z_axis = points[2, :]
        self.z_origin = origin[2]

        origin = origin[:2, :].astype(np.int32)
        points = points[:2, :].astype(np.int32)

        diff = np.subtract(points, origin).astype(np.int32)
        diff_z = np.subtract(self.z_axis, self.z_origin)

        nmbr_samples = points.shape[1]

        self.d_fast = np.empty([1, nmbr_samples], dtype=np.int32)
        self.d_slow = np.empty([1, nmbr_samples], dtype=np.int32)

        self.step_diag = np.copysign(
            np.ones([2, nmbr_samples], dtype=np.int32), diff[:2, :]
        )
        self.step_diag = self.step_diag.astype(np.int32, copy=True)

        mask = np.greater_equal(np.abs(diff[0, :]), np.abs(diff[1, :])).reshape(
            1, nmbr_samples
        )

        np.abs(diff[0, :], out=self.d_fast, where=mask[0, :])
        np.abs(diff[1, :], out=self.d_slow, where=mask[0, :])

        self.step_par = np.zeros([2, nmbr_samples], dtype=np.float32)
        np.copysign(
            np.ones(1, dtype=np.int32),
            diff[0, :],
            out=self.step_par[0, :],
            where=mask[0, :],
        )
        np.copysign(
            np.ones(1, dtype=np.int32),
            diff[1, :],
            out=self.step_par[1, :],
            where=~mask[0, :],
        )
        self.step_par = self.step_par.astype(np.int32, copy=True)

        np.abs(diff[1, :], out=self.d_fast, where=~mask[0, :])
        np.abs(diff[0, :], out=self.d_slow, where=~mask[0, :])

        self.origin = np.int32(origin)
        self.points = points[:3, :]

        # now determine the evolution of the line for every incremental step along the dominant direction
        self.d = diff
        self.vertical_step = np.divide(diff_z, self.d_fast)

    def cast(self, roi=None):
        rays_array = np.broadcast_to(self.origin, self.points.shape).copy()
        z_values = np.broadcast_to(self.z_origin, [1, self.points.shape[1]]).copy()

        active = np.ones([1, rays_array.shape[1]]).astype(bool)
        loss = np.divide(self.d_fast, 2)

        if roi is not None:
            # lower-limit for roi
            lower_limit = self.origin[:2, 0] - roi
            # higher limit roi
            upper_limit = self.origin[:2, 0] + roi

        # start generator loop
        if self.d_fast.size <= 1:
            return
        for step in range(int(np.max(self.d_fast))):
            np.subtract(loss, self.d_slow, out=loss, where=active)

            np.add(
                rays_array[:2],
                self.step_diag,
                out=rays_array[:2],
                where=loss < 0,
            )
            np.add(
                rays_array[:2],
                self.step_par,
                out=rays_array[:2],
                where=loss >= 0,
            )

            np.add(z_values, self.vertical_step, out=z_values, where=active)

            # correct loss for all diagonal steps
            np.add(loss, self.d_fast, out=loss, where=loss < 0)

            # np.logical_or(rays_array[0,:]>=0, rays_array[1,:]<0, out=active)
            active[self.d_fast <= step] = False
            if roi is not None:
                # deactivate rays which leave roi
                roi_mask = (
                    np.logical_or(
                        (rays_array[:2, :] <= lower_limit[:, np.newaxis]).any(axis=0),
                        (rays_array[:2, :] >= upper_limit[:, np.newaxis]).any(axis=0),
                    ),
                )
                active[:, roi_mask[0]] = False
            if not active.any():
                break
            yield (
                rays_array[:, active[0, :]].astype(np.int32),
                z_values[active],
            )


class GridMap(object):
    """
    Docstring TODO
    """

    MAX_SIZE = 10000

    def __init__(
        self,
        cell_size=0.1,
        *,
        height=None,
        width=None,
        sensor_range_u=[20, 20, 0],
        max_height=5.0
    ):
        if height is not None:
            self.height = height
        else:
            self.height = int(2 * sensor_range_u[0] // cell_size)
        if width is not None:
            self.width = height
        else:
            self.width = int(2 * sensor_range_u[1] // cell_size)

        self.cell_size = cell_size  # [m]

        # dummy value for now
        # pos of sensor in grid
        self.sensor_origin = [
            int(self.height),
            int(self.width),
            int(2 / 0.05),
        ]

        self.difference = Isometry()

        # vehicle_T_map of the current pose
        self.pose = Isometry()  # vehicle pose in world
        self.trajectory = []  # [world coordinates] from previous frames

        ## transformations used in refactored class
        # we need:
        # world_T_grid: grid origin world frame
        # world_T_vehicle: vehicle in world
        self.sensor_range_u = np.array([sensor_range_u[0], sensor_range_u[1], 0])  # [m]
        self.world_T_grid = Isometry(
            translation=-self.sensor_range_u.astype(np.float32)
        )
        self.world_T_vehicle = Isometry()
        self.grid_T_vehicle = self.world_T_grid.inverse() @ self.world_T_vehicle

        self.max_height = max_height
        # channnels: occupation, intensity, height, r, b, g, cnt
        self.initial_grid = np.array([0, 0, self.max_height, 0, 255, 0, 0])
        self.grid_u = np.full(
            [self.height, self.width, 3],
            self.initial_grid[:3],
            dtype=np.float32,
        )
        self.grid_rgb = np.full(
            [self.height, self.width, 3],
            self.initial_grid[3:-1],
            dtype=np.uint8,
        )
        # the minimum range which is plotted around the vehicle
        self.marker = []
        self.arrows = []
        self.boundaries = []

        self.cameras = []

    def veh_to_grid(self, points: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.floor_divide(self.grid_T_vehicle @ points, self.cell_size), 0
        ).astype(np.int32)

    def world_to_grid(self, points: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.floor_divide(self.world_T_grid.inverse() @ points, self.cell_size),
            0,
        ).astype(np.int32)

    def update_transformations(self, world_T_vehicle):

        self.world_T_vehicle = world_T_vehicle

        if len(self.trajectory) == 0:
            # initialize at first update
            # self.world_T_grid = self.world_T_vehicle @ self.world_T_grid
            self.world_T_grid.translation += world_T_vehicle.translation

        self.grid_T_vehicle = self.world_T_grid.inverse() @ self.world_T_vehicle
        if world_T_vehicle not in self.trajectory:
            self.trajectory.append(world_T_vehicle)

    def draw_frame(self, world_T_frame: Isometry):
        points = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        fpoints = world_T_frame @ points
        self.arrows.append(fpoints)

    def update_u(
        self,
        point_cloud: np.ndarray,
        veh_T_sensor: Isometry = Isometry(),
        world_T_veh: Isometry = Isometry(),
    ):
        # updating includes the following steps:
        # - 1. update sensor_origin and vehicle position
        # - 2. grow the grid if needed
        # - 3. transform all points into grid coordinates
        # - 4. cast rays for every point
        # - 5. update all cells

        # 1.
        self.update_transformations(world_T_veh)

        grid_T_sensor = self.grid_T_vehicle @ veh_T_sensor

        # 2.
        # first pad map
        self.pad_grid(grid_T_sensor)
        # all positions have to be adjusted for left padding
        # self.world_T_grid.translation += shift * self.cell_size

        # frame_test = Isometry(translation=world_T_veh.translation, rotation=Quaternion(axis=[0,0,1], degrees=45))
        self.draw_frame(world_T_frame=self.world_T_vehicle)
        grid_T_sensor = self.grid_T_vehicle @ veh_T_sensor

        # 3. dtype float so that height is preserved. is later casted to int in RayArray
        grid_points = self.veh_to_grid(point_cloud[:3, :]).astype(np.float32)

        # now filter the point cloud
        # remove all points above max_height
        # remove points outside of the grid
        greater_zero = (grid_points[:2, :] >= 0).all(axis=0)
        filter_roi = np.logical_and.reduce(
            [
                greater_zero,
                grid_points[0, :] < self.height,
                grid_points[1, :] < self.width,
            ]
        )

        filtered_grid_points = grid_points[:, filter_roi]
        filtered_pc = point_cloud[:, filter_roi]
        assert filtered_grid_points.shape[1] == filtered_pc.shape[1]
        filtered_grid_points[2, :] = np.maximum(0.0, filtered_pc[2, :])

        # 4.
        sensor_origin_grid = grid_T_sensor.translation
        np.floor_divide(
            sensor_origin_grid[:2], self.cell_size, out=sensor_origin_grid[:2]
        )
        ray_array = RayArray(origin=sensor_origin_grid, points=filtered_grid_points)

        filtered_grid_points = filtered_grid_points.astype(np.int32)
        # occupation status
        self.grid_u[filtered_grid_points.T[:, 0], filtered_grid_points.T[:, 1], 0] = 1.0

        # intensity
        self.grid_u[
            filtered_grid_points.T[:, 0], filtered_grid_points.T[:, 1], 1
        ] = filtered_pc[3, :]

        sensor_range_grid = np.floor_divide(self.sensor_range_u[:2], self.cell_size)
        for (indeces, height) in ray_array.cast(roi=sensor_range_grid):
            if (indeces == 0).all():
                logging.warn(
                    "no valid indeces from ray casting. proceeding to next iteration"
                )
                continue

            # remove from indeces
            indeces = np.transpose(indeces)

            # filter indeces
            # outside of grid
            valid = np.logical_and.reduce(
                [
                    (indeces[:, :2] >= 0).all(axis=1),
                    (indeces[:, 0] < self.grid_u.shape[0]),
                    (indeces[:, 1] < self.grid_u.shape[1]),
                ]
            )
            indeces = indeces[valid]
            height = height[valid]

            self.grid_u[indeces[:, 0], indeces[:, 1], 2] = np.minimum(
                height,
                self.grid_u[indeces[:, 0], indeces[:, 1], 2],
            )
            # self.grid_u[indeces[:, 0], indeces[:, 1], 2] = 0.0

    def update_bev(self, camera: BirdsEyeView):
        hom_T = self.grid_T_vehicle.matrix
        transform = np.array(
            [
                [hom_T[0, 0], hom_T[0, 1], hom_T[0, 3]],
                [hom_T[1, 0], hom_T[1, 1], hom_T[1, 3]],
                [0, 0, 1],
            ]
        )

        bev_veh = camera.transform(
            offset=(0, 0),
            resolution=self.cell_size,
            out_size=(self.width, self.height),
            aux_transform=transform,
            borderValue=(0, 255, 0),
        )

        center = self.veh_to_grid(np.array([0, 0, 0]))
        look_ahead = 10 // self.cell_size
        row_low = int(max(0, center[0] - look_ahead))
        row_heigh = int(min(self.height, center[0] + look_ahead))

        column_low = int(max(0, center[1] - look_ahead))
        column_heigh = int(min(self.width, center[1] + look_ahead))

        """
        print(row_low)
        print(row_heigh)
        print(column_low)
        print(column_heigh)
        """
        mask = (bev_veh != (0, 255, 0)).all(axis=-1)  # , keepdims=True)

        distance_mask = np.zeros_like(mask, dtype=np.bool)
        distance_mask[row_low:row_heigh, column_low:column_heigh] = True

        combined_mask = np.logical_and(mask, distance_mask)
        # combined_mask = mask
        stitcher = Stitcher()

        self.grid_rgb[combined_mask] = bev_veh[combined_mask]

    def render(self, debug: bool = True):
        # render for now only the occupation status
        # remove all infs from height channel

        if debug:
            # if debuggin active, return a rgb image for all channels with all markers and frames

            # first normalize height

            height = np.multiply(
                255, np.divide(self.grid_u[:, :, 2:3], self.max_height)
            ).astype(np.uint8)
            height = cv2.applyColorMap(height, cv2.COLORMAP_PLASMA)

            # occupation is binary and doesn't have to be normalized
            occupation = cv2.cvtColor(
                np.multiply(self.grid_u[:, :, 0], 255).astype(np.uint8),
                cv2.COLOR_GRAY2BGR,
            )

            # intensity
            # find argmax in intensity channel
            max_intensity = np.max(self.grid_u[:, :, 1])
            intensity = np.multiply(
                np.divide(self.grid_u[:, :, 1], max_intensity), 255
            ).astype(np.uint8)
            intensity = cv2.applyColorMap(intensity, cv2.COLORMAP_JET)

            # now draw all markers and frames for debugging
            for boundary in self.boundaries:

                grid_bound = self.grid_T_vehicle @ boundary
                px_grid_bounds = np.floor_divide(grid_bound, self.cell_size).astype(
                    np.int32
                )
                px_grid_bounds = np.array([px_grid_bounds[1, :], px_grid_bounds[0, :]])
                px_grid_bounds = np.swapaxes(px_grid_bounds, 0, 1)

                height = cv2.polylines(
                    height,
                    px_grid_bounds[np.newaxis, :, :],
                    False,
                    color=(0, 0, 0),
                    thickness=3,
                )
                occupation = cv2.polylines(
                    occupation,
                    px_grid_bounds[np.newaxis, :, :],
                    False,
                    color=(0, 255, 0),
                    thickness=3,
                )
                intensity = cv2.polylines(
                    intensity,
                    px_grid_bounds[np.newaxis, :, :],
                    False,
                    color=(0, 0, 0),
                    thickness=3,
                )

            for m in self.marker:
                (x, y) = np.squeeze(self.world_to_grid(m))[:2]
                height = cv2.drawMarker(
                    height,
                    (y, x),
                    color=(0, 0, 0),
                    markerType=cv2.MARKER_SQUARE,
                    markerSize=5,
                )
                occupation = cv2.drawMarker(
                    occupation,
                    (y, x),
                    color=(0, 255, 0),
                    markerType=cv2.MARKER_SQUARE,
                    markerSize=5,
                )
                intensity = cv2.drawMarker(
                    intensity,
                    (y, x),
                    color=(0, 0, 0),
                    markerType=cv2.MARKER_SQUARE,
                    markerSize=5,
                )

            for points in self.arrows:
                grid_points = self.world_to_grid(points)
                origin = (grid_points[1, 0], grid_points[0, 0])

                height = cv2.line(
                    height,
                    origin,
                    (grid_points[1, 1], grid_points[0, 1]),
                    color=(255, 255, 255),
                    thickness=3,
                )
                intensity = cv2.line(
                    intensity,
                    origin,
                    (grid_points[1, 1], grid_points[0, 1]),
                    color=(255, 255, 255),
                    thickness=3,
                )
                occupation = cv2.line(
                    occupation,
                    origin,
                    (grid_points[1, 1], grid_points[0, 1]),
                    color=(0, 255, 0),
                    thickness=3,
                )

                height = cv2.line(
                    height,
                    origin,
                    (grid_points[1, 2], grid_points[0, 2]),
                    color=(0, 0, 0),
                    thickness=3,
                )
                occupation = cv2.line(
                    occupation,
                    origin,
                    (grid_points[1, 2], grid_points[0, 2]),
                    color=(0, 255, 0),
                    thickness=3,
                )
                intensity = cv2.line(
                    intensity,
                    origin,
                    (grid_points[1, 2], grid_points[0, 2]),
                    color=(0, 0, 0),
                    thickness=3,
                )

                vertices = self.get_roi_vertices(
                    roi=(self.sensor_range_u[0], self.sensor_range_u[1])
                )
                # transform vertices to image coords for cv
                vertices = np.column_stack((vertices[:, 1], vertices[:, 0]))
                # TODO decent colors
                height = cv2.polylines(
                    height, [vertices], isClosed=True, color=(255, 255, 255)
                )
                occupation = cv2.polylines(
                    occupation, [vertices], isClosed=True, color=(0, 255, 0)
                )
                intensity = cv2.polylines(
                    intensity, [vertices], isClosed=True, color=(0, 255, 0)
                )

            return np.concatenate([occupation, intensity, height], axis=2)
        else:
            return self.grid_u

    def get_roi_vertices(self, roi):
        from shapely import affinity
        from shapely.geometry import box

        # define the roi as a box

        roi_box_veh = box(
            -roi[0],
            -roi[1],
            roi[0],
            roi[1],
        )
        # transform this box into grid coordinate
        roi_box_veh = affinity.translate(
            roi_box_veh,
            xoff=self.grid_T_vehicle.translation[0],
            yoff=self.grid_T_vehicle.translation[1],
        )
        roi_box_veh = affinity.rotate(
            roi_box_veh,
            angle=self.grid_T_vehicle.rotation.yaw_pitch_roll[0],
            use_radians=True,
        )
        vertices = np.array(roi_box_veh.exterior.coords)
        vertices = np.floor_divide(vertices, self.cell_size).astype(np.int32)
        # return vertices
        return vertices

    def pad_grid(self, grid_T_sensor: Isometry):
        # now build a box (4 vertices) reporesenting the roi of the sensor
        roi_ = np.array(
            [
                [
                    -self.sensor_range_u[0],
                    -self.sensor_range_u[0],
                    self.sensor_range_u[0],
                    self.sensor_range_u[0],
                ],
                [
                    -self.sensor_range_u[1],
                    self.sensor_range_u[1],
                    -self.sensor_range_u[1],
                    self.sensor_range_u[1],
                ],
                [0, 0, 0, 0],
            ]
        )

        sensor_roi_grid = np.floor_divide(
            (grid_T_sensor @ roi_), self.cell_size
        ).astype(np.int32)

        left = np.abs(
            np.minimum(
                0,
                [
                    np.min(sensor_roi_grid[0, :]),
                    np.min(sensor_roi_grid[1, :]),
                ],
            )
        ).astype(np.int32)
        right = np.maximum(
            0,
            [
                np.max(sensor_roi_grid[0, :]) - self.height,
                np.max(sensor_roi_grid[1, :]) - self.width,
            ],
        ).astype(np.int32)

        # if x_low > 0 or x_high > 0 or y_low > 0 or y_high > 0:
        if np.logical_or(left[:2] > 0, right[:2] > 0).any():
            # only pad until max size
            new_grid_size = [
                min(left[0] + right[0] + self.height, self.MAX_SIZE),
                min(left[1] + right[1] + self.width, self.MAX_SIZE),
            ]
            cp_grid = np.full(
                [new_grid_size[0], new_grid_size[1], 3],
                self.initial_grid[:3],
            )
            cp_grid_rgb = np.full(
                [new_grid_size[0], new_grid_size[1], 3],
                self.initial_grid[3:-1],
                dtype=np.uint8,
            )
            # compute the grid borders with the given paddings:
            # if the grid can be expanded the margin is self.height
            # but if there has been hit a boundary, we only want to insert MAX_SIZE - (left+right)
            margin = new_grid_size[:2] - left[:2] - right[:2]

            # for now handle x and y expansion seperatly
            # is x expansion possible?
            if new_grid_size[0] < self.MAX_SIZE:
                x_min = 0
                x_max = self.height
                offset_x = -left[0]  # checked
            # if not compute margins
            else:
                # only handle the dominant expansion (should practically never happen given sensible parameters)
                if left[0] < right[0]:
                    left[0] = 0
                else:
                    right[0] = 0

                if left[1] < right[1]:
                    left[1] = 0
                else:
                    right[1] = 0

                # if left padding
                if left[0] != 0:
                    x_min = 0
                    x_max = new_grid_size[0] - left[0]
                    offset_x = -left[0]  # checked
                # if right_padding
                elif right[0] != 0:
                    margin = new_grid_size[0] - right[0]
                    x_max = self.height
                    x_min = self.height - margin
                    offset_x = x_min  # checked
                else:
                    x_min = 0
                    x_max = self.height
                    offset_x = 0

            # is y expansion possible?
            if new_grid_size[1] < self.MAX_SIZE:
                y_min = 0
                y_max = self.width
                offset_y = -left[1]  # checked

            # if not compute margins
            else:
                # if left padding
                if left[1] != 0:
                    y_min = 0
                    y_max = new_grid_size[1] - left[1]
                    offset_y = -left[1]  # TODO
                # if right_padding
                elif right[1] != 0:
                    margin = new_grid_size[1] - right[1]
                    y_max = self.width
                    y_min = self.width - margin
                    offset_y = y_min  # TODO
                else:
                    y_min = 0
                    y_max = self.width
                    offset_y = 0

            if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
                logging.warn(
                    "Unable to pad grid; please look to debugging logs for further informations"
                )
                logging.debug(
                    "left: %s, right %s, x_min: %s, x_max: %s, y_min: %s, y_max: %s"
                    % (left, right, x_min, x_max, y_min, y_max)
                )
                return

            cp_grid[
                left[0] : left[0] + (x_max - x_min),
                left[1] : left[1] + (y_max - y_min),
                :,
            ] = self.grid_u[x_min:x_max, y_min:y_max, :]

            self.grid_u = cp_grid.copy()

            cp_grid_rgb[
                left[0] : left[0] + (x_max - x_min),
                left[1] : left[1] + (y_max - y_min),
                :,
            ] = self.grid_rgb[x_min:x_max, y_min:y_max, :]

            self.grid_rgb = cp_grid_rgb.copy()

            (self.height, self.width, _) = self.grid_u.shape
            logging.debug("Pad grid")
            logging.debug(self.grid_u.shape)

            # if simply shifted left the grid is ranslated in positiv direction
            # if there was a left shift: adjust the translation by the margin, that the map has been shifted (right)

            # note: the shift is happening in grid-frame so we have to transform first
            offset = [
                offset_x * self.cell_size,
                offset_y * self.cell_size,
                0,
            ]
            np.add(
                self.world_T_grid.translation[:2],
                offset[:2],
                out=self.world_T_grid.translation[:2],
            )
            self.grid_T_vehicle = self.world_T_grid.inverse() @ self.world_T_vehicle
            logging.debug(
                "shift grid by %s, %s"
                % (offset_x * self.cell_size, offset_y * self.cell_size)
            )

    def draw_marker(self, world_point):
        """
        add the world coord of a marker to the marker list
        """
        self.marker.append(world_point)

    def vehicleToGridPoint(self, vehicle_point) -> np.ndarray:
        # transform a vehicle point to a discrete point on the grid
        grid_point = np.floor_divide(
            self.difference @ vehicle_point, self.cell_size
        ) + [0, self.width // 2, 0]
        return grid_point

    def deltaPose(self, pose):
        self.pose = pose
        if len(self.trajectory) == 0:
            self.trajectory.append(self.pose)
        if pose != self.trajectory[-1]:
            self.trajectory.append(self.pose)
            # compute offset from map origin
            self.difference = self.trajectory[0].inverse() @ self.pose

    def saveGrid(self, path: Path):
        # save using pickle
        # print(self.__dict__)
        with open(str(path), "wb+") as f:
            pickle.dump(self.__dict__, f, -1)

    @classmethod
    def loadGrid(cls, path: Path):
        with open(str(path), "rb+") as f:
            file_dict = pickle.load(f)
        instance = cls()
        instance.__dict__ = file_dict
        return instance

    @staticmethod
    def colormap(value):
        """
        Convert the value to a color value
        """
        # TODO: implementation
        return (0, 0, 0)

    def __str__(self):
        string = "GridMap Object"
        string += " \n"
        string += "[%s, %s] Cells at %s resolution" % (
            self.height,
            self.width,
            self.cell_size,
        )
        string += " \n"
        string += "Sensor origin at " + str(self.sensor_origin)
        return string


if __name__ == "__main__":
    # load a point cloud
    # PATH = "/home/dominic/data/master_thesis/lyft/level5_lidar/scene_0/LIDAR_TOP_1553294662300940.pcd"
    from math import acos, asin, pi

    grid = GridMap()
    nmbr_samples = 2000
    radius = 20
    synth_data = np.zeros([4, nmbr_samples])
    for i, rad in enumerate(0, 2 * pi, 2 * pi / nmbr_samples):
        synth_data[0, i] = rad * acos(rad)
        synth_data[1, i] = rad * asin(rad)

    grid.update_u(
        point_cloud=synth_data,
        veh_T_sensor=Isometry(translation=[0, 0, 5]),
        world_T_veh=Isometry(translation=[100, 100, 0]),
    )

    """
    PATH = "/home/dominic/data/master_thesis/lyft/level5_lidar/scene_0/LIDAR_TOP_1553294662300940.pcd"

    parser = PCDParser()
    point_cloud, viewpoint = parser.read(PATH)
    # create grid
    grid = GridMap()
    grid.update(point_cloud=point_cloud, sensor_position=viewpoint[:3])
    grid.plotMap()
    path = Path("test_map.pkl")
    grid.saveGrid(path)
    load_grid = GridMap.loadGrid(path)
    print(load_grid)
    print(grid)
    """
