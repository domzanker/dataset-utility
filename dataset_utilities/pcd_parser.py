import numpy as np
from dataset_utilities.transformation import Isometry
from pathlib import Path


class PCDParser:

    """Docstring for PCDParser. """

    def __init__(self):
        self.version = [0.7]
        self.fields = ["x", "y", "z"]
        self.size = []
        self.type = None
        self.count = None
        self.width = None
        self.height = None
        self.viewpoint = [0, 0, 0, 1, 0, 0, 0]
        self.points = None
        self.data = "ascii"

        self.delimiter = " "

    def composeHeader(self) -> str:
        header_dict = {
            "VERSION": self.version,
            "FIELDS": self.fields,
            "SIZE": self.size,
            "TYPE": self.type,
            "COUNT": self.count,
            "WIDTH": self.width,
            "HEIGHT": self.height,
            "VIEWPOINT": self.viewpoint,
            "POINTS": self.points,
            "DATA": self.data,
        }

        header = "# .PCD v.7 - Point Cloud Data file format \n"
        for key, value in header_dict.items():
            header = header + key + self.delimiter
            for x in value:
                header = header + str(x) + self.delimiter
            header = header + "\n"
        return header

    def write(self, point_cloud: np.ndarray, file_name: str, binary: bool = False):
        ## PointCloud has to be shape [nPoints, channels]
        try:
            assert point_cloud.shape[1] in [3, 4]
            assert point_cloud.shape[0] > 0
        except Exception as e:
            print(point_cloud.shape)
            raise e

        FIELDS = ["x", "y", "z", "intensity"]
        self.fields = [FIELDS[i] for i in range(point_cloud.shape[1])]

        self.size = []
        self.type = []
        self.count = []
        for dim in point_cloud[0, :]:
            self.size.append(point_cloud.itemsize)
            if isinstance(dim, np.int8):
                self.type.append("I")
            elif isinstance(dim, np.float32):
                self.type.append("F")
            else:
                raise TypeError
            self.count.append(1)

        self.width = [point_cloud.shape[0]]
        self.height = [1]
        self.points = [point_cloud.shape[0]]
        if binary:
            self.data = ["binary"]
            raise NotImplementedError
        else:
            self.data = ["ascii"]

        header = self.composeHeader()
        with Path(file_name).with_suffix(".pcd").open("w+") as f:
            f.write(header)
            for i in range(0, self.points[0]):
                point = point_cloud[i, :]
                line = ""
                for p in point:
                    line = line + str(p) + self.delimiter
                f.write(line + "\n")

    def addCalibration(self, calibration: Isometry = Isometry):
        self.viewpoint = []
        for translation in calibration.translation:
            self.viewpoint.append(translation)

        for rotation in calibration.rotation:
            self.viewpoint.append(rotation)

        assert len(self.viewpoint) == 7

    def read(self, file: str):
        header = self.readHeader(file)
        nmbr_elem_per_point = len(header["FIELDS"])
        nmbr_of_points = int(header["POINTS"])
        if header["DATA"] == "binary":
            binary = True
            raise NotImplementedError
        else:
            binary = False

        if not binary:
            point_cloud = np.zeros([nmbr_elem_per_point, nmbr_of_points])
            i = 0
            with open(file, "r") as f:
                # TODO read header
                for line in f:
                    if line.split(self.delimiter)[0] == "DATA":
                        # begin reading from here
                        for i in range(nmbr_of_points):
                            point_cloud[:, i] = next(f).split(self.delimiter)[:-1]

            return (
                np.asarray(point_cloud),
                np.asarray(header["VIEWPOINT"], dtype=np.float32),
            )

    def readHeader(self, file):
        header_dict = {}
        with open(file, "r") as f:
            for line in f.readlines():
                if line[0] == "#":
                    continue
                line_sep = line.split(self.delimiter)
                if len(line_sep[1:-1]) > 1:
                    header_dict[str(line_sep[0])] = line_sep[1:-1]
                else:
                    header_dict[str(line_sep[0])] = line_sep[1:-1][0]

                if line_sep[0] == "DATA":
                    break
        return header_dict


if __name__ == "__main__":
    PATH = "/home/dominic/data/master_thesis/lyft/level5_lidar/scene_0/LIDAR_TOP_1553294662300940.pcd"
    parser = PCDParser()
    point_cloud = parser.read(file=PATH)
    print(point_cloud.shape)
