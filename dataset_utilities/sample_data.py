from pathlib import Path
import h5py
import numpy as np
import cv2


class SampleData:
    def __init__(
        self,
        scene: int,
        sample: int,
        base_path: Path,
        *,
        includes_debug=False,
        prefix=None
    ):

        self.scene = scene
        self.sample = sample
        self.base_path = base_path
        self.include_debug = includes_debug

        self._build_directory_structure(scene, sample)

        self.rgb = None
        self.lidar = None

        self.distance_map = None
        self.direction_map = None
        self.end_points = None

        self.debug_data = {}
        self.debug_images = {}

        self.prefix = "" if prefix is None else prefix + "_"

    def _build_directory_structure(self, scene: int, sample: int):
        self.scene_path = self.base_path / ("scene_%s" % scene)
        self.sample_data_path = self.base_path / "data"

        self.scene_path.mkdir(exist_ok=True, parents=True)
        self.sample_data_path.mkdir(exist_ok=True, parents=True)

        if self.include_debug:
            self.sample_debug_path = self.scene_path / ("sample_%s" % sample)
            self.sample_debug_path.mkdir(exist_ok=True, parents=True)

    def add_data(self, rgb: np.ndarray, lidar: np.ndarray):
        assert rgb.ndim == lidar.ndim
        assert np.alltrue(rgb.shape[:2] == lidar.shape[:2])

        self.rgb = rgb[:, :, ::-1]
        self.lidar = lidar

    def add_targets(
        self,
        direction_map: np.ndarray,
        distance_map: np.ndarray,
        end_points: np.ndarray,
        ground_truth: np.ndarray,
    ):
        assert self.rgb.ndim == direction_map.ndim
        assert self.rgb.ndim == distance_map.ndim
        assert self.rgb.ndim == end_points.ndim
        assert self.rgb.ndim == ground_truth.ndim

        assert np.alltrue(self.rgb[:, :, 0].shape == direction_map[:, :, 0].shape)
        assert np.alltrue(self.rgb[:, :, 0].shape == distance_map[:, :, 0].shape)
        assert np.alltrue(self.rgb[:, :, 0].shape == end_points[:, :, 0].shape)
        assert np.alltrue(self.rgb[:, :, 0].shape == ground_truth[:, :, 0].shape)

        self.distance_map = distance_map
        self.direction_map = direction_map
        self.end_points = end_points
        self.ground_truth = ground_truth

    def add_debug_image(self, image: np.ndarray, name_tag: str):
        assert image.shape[2] in (1, 3)

        self.debug_images[str(name_tag)] = image.astype(np.uint8)

    def write(self):
        sample_data_file = self.sample_data_path / (
            "%sscene_%s_sample_%s_data.h5" % (self.prefix, self.scene, self.sample)
        )
        with h5py.File(str(sample_data_file), "w", swmr=True) as h:

            # add rgb
            rgb = self.rgb.astype(np.uint8)
            h.create_dataset("rgb", data=rgb)

            # add height
            height = self.lidar[:, :, 2]
            height = height - height.min()
            height = height / (height.max() + 1e-12)
            height = height.astype(np.float16)
            h.create_dataset("lidar_height", data=height)

            # add distance
            distance = self.distance_map
            distance = distance - distance.min()
            distance = distance / (distance.max() + 1e-12)
            distance = distance.astype(np.float16)
            h.create_dataset("inverse_distance_map", data=distance)

            # add end points
            end = self.end_points
            end = end - end.min()
            end = end / (end.max() + 1e-12)
            end = end.astype(np.float16)
            h.create_dataset("end_points_map", data=end)

            # add direction
            direction = self.direction_map
            direction = direction / (
                np.abs(np.linalg.norm(direction, axis=2, keepdims=True)) + 1e-12
            )
            direction = direction.astype(np.float32)
            h.create_dataset("road_direction_map", data=direction)

        if not self.include_debug:
            return
        for key, value in self.debug_images.items():
            debug_file = self.sample_debug_path / ("%s.png" % key)
            cv2.imwrite(str(debug_file), value)
