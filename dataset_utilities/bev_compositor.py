"""
File: bev_compositor.py
Author: Dominic Zanker
Email: yourname@email.com
Github: https://github.com/yourname
Description:
    generate a BEV image by compositing several cameras
"""
from dataset_utilities.camera import Camera, BirdsEyeView, Lidar, SensorBase
from dataset_utilities.transformation import Isometry
import numpy as np
import cv2
import logging
import shapely.geometry


def normalizeImage(img):
    # convert to hsv
    img = (img).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = cv2.extractChannel(hsv, 0)
    normed = cv2.equalizeHist(l_channel)
    norm_img = cv2.insertChannel(normed, hsv, 0)
    return cv2.cvtColor(norm_img, cv2.COLOR_LAB2BGR)


class BEVCompositor(object):

    """Docstring for BEVCompositor. """

    def __init__(self, resolution: float = 0.04, reach: list = [10.0, 5.0]):
        """
        resolution -- resolution of BEVComposition [m/px]
        reach -- look ahead for ROI. [x,y] in Vehicle frame
        """
        self.resolution = resolution  # [m/px]

        self.reach = reach
        self.map_width = 2 * int(reach[1] // self.resolution)
        self.map_height = 2 * int(reach[0] // self.resolution)
        self.map = np.zeros([self.map_height, self.map_width, 3])
        self.pixel_mask = np.ones_like(self.map[:, :, 0], dtype=np.bool)
        self.cnt = np.zeros_like(self.map[:, :, 0])

        self.label = np.full([self.map_height, self.map_width, 1], 255, dtype=np.uint8)

        self.boundaries = []

        # sensors
        self.sensors = {}
        # old maps with ego_poses as key
        self.maps = {}
        self.views = {}

    def addSensor(self, sensor: SensorBase):
        """ add a new Sensor to the Compositor """
        logging.debug("add new Sensor: %s" % sensor.id)
        if sensor.sensor_type == "lidar":
            self.sensors[sensor.id] = sensor
        elif sensor.sensor_type == "bev":
            self.sensors[sensor.id] = sensor

    def composeImage(self, debug=False):
        self.map = np.zeros([self.map_height, self.map_width, 3])
        self.pixel_mask = np.ones_like(self.map[:, :, 0], dtype=np.bool)
        self.cnt = np.zeros_like(self.map[:, :, 0])

        for (key, sensor) in self.sensors.items():
            if isinstance(sensor, Camera):
                b_ = sensor.transform(
                    offset=[self.reach[1], self.reach[0]],
                    resolution=self.resolution,
                    out_size=(self.map_width, self.map_height),
                )
                self.views[sensor] = (b_, sensor.extrinsic)

                m_ = sensor.transform(
                    np.ones_like(sensor.data[:, :, 0]),
                    offset=[self.reach[1], self.reach[0]],
                    resolution=self.resolution,
                    out_size=(self.map_width, self.map_height),
                ).astype(np.bool)

                m_[(b_ == (0, 0, 0)).all(axis=2)] = False

                self.pixel_mask = np.logical_or(self.pixel_mask, m_)

                np.add(self.cnt, 1, out=self.cnt, where=m_)
                np.add(self.map, b_, out=self.map, where=m_[:, :, np.newaxis])

        # self.pixel_mask[self.pixel_mask == 0] = 1
        self.cnt[self.cnt == 0] = 1
        np.divide(
            self.map,
            self.cnt[:, :, np.newaxis],
            out=self.map,
            where=self.pixel_mask[:, :, np.newaxis],
        ).astype(np.int32)

        if debug:
            self.map = cv2.drawMarker(
                self.map,
                (self.map_width // 2, self.map_height // 2),
                color=(255, 0, 0),
            )

            self.map[(self.label == 0).all(axis=2)] = self.label[
                (self.label == 0).all(axis=2)
            ]

        return self.map

    def render_label(self, road_boundaries):
        self.label = np.full(
            [self.map.shape[0], self.map.shape[1], 1], 255, dtype=np.uint8
        )
        if len(road_boundaries) == 2:
            # assume inner and exterior hull
            exterior = road_boundaries[0]
            interior = road_boundaries[1]

            for polyline in exterior:

                polyline = np.swapaxes(np.asarray(polyline), 1, 0)
                polyline = polyline[:, 1:]
                assert polyline.shape[0] in (2, 3)
                polyline = np.floor_divide(
                    polyline[:2, :] + [[self.reach[0]], [self.reach[1]]],
                    self.resolution,
                ).astype(np.int32)
                polyline = np.array([polyline[1, :], polyline[0, :]])
                polyline = np.swapaxes(polyline, 0, 1)
                self.label = cv2.polylines(
                    self.label,
                    polyline[np.newaxis, :, :],
                    False,
                    color=0,
                )
            for polyline in interior:

                polyline = np.swapaxes(np.asarray(polyline), 1, 0)
                polyline = polyline[:, 1:]
                assert polyline.shape[0] in (2, 3)
                polyline = np.floor_divide(
                    polyline[:2, :] + [[self.reach[0]], [self.reach[1]]],
                    self.resolution,
                ).astype(np.int32)
                polyline = np.array([polyline[1, :], polyline[0, :]])
                polyline = np.swapaxes(polyline, 0, 1)
                self.label = cv2.polylines(
                    self.label,
                    polyline[np.newaxis, :, :],
                    True,
                    color=0,
                )

        else:
            for polyline in road_boundaries:
                assert polyline.shape[0] in (2, 3)
                polyline = np.floor_divide(
                    polyline[:2, :] + [[self.reach[0]], [self.reach[1]]],
                    self.resolution,
                ).astype(np.int32)
                polyline = np.array([polyline[1, :], polyline[0, :]])
                polyline = np.swapaxes(polyline, 0, 1)
                self.label = cv2.polylines(
                    self.label,
                    polyline[np.newaxis, :, :],
                    False,
                    color=0,
                )
        # self.label = np.flipud(self.label)


if __name__ == "__main__":

    from lyft_dataset_sdk.lyftdataset import LyftDataset
    import os
    import pyquaternion

    # test BEVCompositor class
    data_path = "~/data/master_thesis/lyft_lvl5"
    level5data = LyftDataset(
        data_path=data_path,
        json_path="~/data/master_thesis/lyft_lvl5/v1.02-train",
        verbose=True,
    )
    # level5data.list_scenes()
    my_scene = level5data.scene[1]
    # for now we are only interested in one sample
    first_sample_token = my_scene["first_sample_token"]
    sample = level5data.get(table_name="sample", token=first_sample_token)

    sample_data = level5data.get(
        table_name="sample_data", token=sample["data"]["CAM_FRONT"]
    )
    ego_pose_token = sample_data["ego_pose_token"]
    # level5data.explorer.render_ego_centric_map(
    #    sample_data["token"], axes_limit=10)

    compositor = BEVCompositor(roi_width=920, roi_height=1280, resolution=0.04)
    # sensor_keys = ['CAM_FRONT', 'CAM_FRONT_RIGHT']
    sensor_keys = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        # "CAM_BACK_LEFT",
        # "CAM_BACK_RIGHT",
        # "CAM_BACK",
    ]
    imgs = {}
    for key in sensor_keys:
        sample_data = level5data.get(
            table_name="sample_data", token=sample["data"][key]
        )
        img_path = os.path.join(data_path, sample_data["filename"])
        calibration = level5data.get(
            table_name="calibrated_sensor",
            token=sample_data["calibrated_sensor_token"],
        )

        i = cv2.imread(img_path)
        imgs[key] = normalizeImage(i)

        rotation = pyquaternion.Quaternion(calibration["rotation"])
        rotation = rotation.unit
        rotation = rotation.inverse
        rotation_matrix = rotation.rotation_matrix

        translation_inv = calibration["translation"]
        translation = rotation_matrix @ np.asarray(translation_inv)

        intrinsic = calibration["camera_intrinsic"]
        compositor.addCamera(
            key,
            dims=imgs[key].shape,
            intrinsic=intrinsic,
            extrinsic_rotation=rotation_matrix,
            extrinsic_translation=translation,
        )

    compositor.composeImage(imgs)
    raw_imgs = compositor.bev

    composition = compositor.returnComposition()
    smooth_composition = cv2.GaussianBlur(composition, (5, 5), 3)
