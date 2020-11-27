import numpy as np
from scipy.linalg import polar
from pyquaternion import Quaternion
import logging
from typing import List
from scipy.spatial.transform import Rotation


def to_homogenous_points(points: np.ndarray):

    try:
        shape = points.shape
    except AttributeError:
        logging.warn(
            "points should allways be provided as numpy arrays not %s" % type(points)
        )
        points.append(1)
        return points

    if points.ndim == 1:
        points = np.expand_dims(points, axis=-1)
        shape = points.shape

    assert shape[0] in [2, 3]
    assert points.ndim == 2

    ones = np.broadcast_to(np.ones(1, dtype=points.dtype), (1, shape[1]))
    return np.vstack((points, ones))


class Isometry:

    """
    Implements a Class for isometric transformations

    The transformation is composed as T * R
    """

    def __init__(
        self,
        translation: np.ndarray = np.zeros(3),
        rotation: Quaternion = Quaternion(),
    ):
        self.translation = np.asarray(translation)
        self.rotation = Quaternion(rotation)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        # TODO assert validity of matrix
        return cls(matrix[:3, 3], Quaternion(matrix=matrix))

    @classmethod
    def from_carla_transform(cls, trafo):
        matrix = np.asarray(trafo.get_matrix())
        # 1. convert the translation to right-handed coordinate system
        rh_translation = np.array([matrix[0, 3], -matrix[1, 3], matrix[2, 3]])

        # 2. convert the rotation to right-handed coordinate system
        yaw = trafo.rotation.yaw
        pitch = trafo.rotation.pitch
        roll = trafo.rotation.roll
        # apply rotation (pitch->yaw->roll)
        # build from euler
        # carla documentation states rotation by YZX
        # https://carla.readthedocs.io/en/latest/python_api/#carla.Rotation
        rotation = Rotation.from_euler("xyz", (roll, -pitch, -yaw), degrees=True)
        (x, y, z, w) = rotation.as_quat()
        rh_quaternion = Quaternion(w, x, y, z)
        return cls(rh_translation, rh_quaternion)

    def translate(self, other: np.ndarray) -> np.ndarray:
        assert other.shape[-1] == 3
        return other + self.translation

    def rotate(self, other: np.ndarray) -> np.ndarray:
        return self.rotate.rotate(other)

    def inverse(self):
        """Return the inverse Isometry"""
        return self.__invert__()

    def invert(self) -> None:
        """Invert the Isometry Object"""
        inv_ = ~self
        self.translation = inv_.translation
        self.rotation = inv_.rotation

    @property
    def matrix(self) -> np.ndarray:
        return self.translation_matrix @ self.rotation_matrix

    # compatibility function when working with carla.Transform
    def get_matrix(self) -> List[List[float]]:
        return self.matrix.tolist()

    # compatibility function when working with carla.Transform
    def get_inverse_matrix(self) -> List[List[float]]:
        return self.inverse().matrix.tolist()

    @matrix.setter
    def matrix(self, matrix):
        t = self.from_matrix(matrix)
        self.translation = t.translation
        self.rotation = t.rotation

    @property
    def rotation_matrix(self):
        return self.rotation.transformation_matrix

    @property
    def translation_matrix(self):
        t = np.array(
            [
                [1, 0, 0, self.translation[0]],
                [0, 1, 0, self.translation[1]],
                [0, 0, 1, self.translation[2]],
                [0, 0, 0, 1],
            ]
        )
        return t

    def transform(self, other) -> np.ndarray:
        """Apply Isometry to other"""
        return self @ other

    @classmethod
    def random(cls):
        """Return a random Isometry"""
        random_rotation = Quaternion.random()
        random_translation = np.random.random(3)
        return cls(translation=random_translation, rotation=random_rotation)

    def __mul__(self, other):
        return self @ other

    def __matmul__(self, other):
        if isinstance(other, Isometry):
            return Isometry.from_matrix(self.matrix @ other.matrix)
        elif isinstance(other, (np.ndarray, list, tuple)):
            other = to_homogenous_points(other)
            prod = self.matrix @ other
            return np.squeeze(prod[:-1] / prod[-1])

    def __invert__(self):
        M_inv = np.eye(4)
        M_inv[:3, :3] = self.rotation.rotation_matrix.T
        M_inv[:3, 3] = -(M_inv[:3, :3] @ self.translation)
        return Isometry.from_matrix(M_inv)

    def __eq__(self, other):
        if isinstance(other, Isometry):
            return (
                self.rotation == other.rotation
                and (self.translation == other.translation).all()
            )
        elif isinstance(other, np.ndarray):
            if other.shape != (4, 4):
                raise ValueError(
                    "Expected Other of shape (4, 4), got %s" % (other.shape)
                )
            return np.array_allclose(self.matrix == other)
        else:
            raise TypeError(
                "Expected Other of type Isometry or numpy.ndarray, got %s" % type(other)
            )

    def __str__(self):
        return str(self.matrix)


class Transformation(Isometry):
    """
    This class just serves as an alias for Isometry at this point
    """

    def __init__(
        self,
        translation: np.ndarray = np.zeros(3),
        rotation: Quaternion = Quaternion(),
    ):
        super().__init__(translation, rotation)


if __name__ == "__main__":
    from math import isclose

    translation = np.random.rand(3)
    rotation = Quaternion.random()
    trafo = Isometry(translation=translation, rotation=rotation)

    print("Testing Isometry class with random Isometry: \n %s" % trafo)

    inv1 = trafo.inverse()
    inv2 = ~trafo

    np.testing.assert_allclose(inv1.matrix(), inv2.matrix())
    print("Inverse: passed")

    t1 = inv1.inverse()
    inv1.invert()
    np.testing.assert_allclose(
        t1.matrix(),
        inv1.matrix(),
        err_msg="invert() and inverse() method are no interchangeable",
    )
    print("Invert: passed")

    T_eye = Isometry()
    print("Indentity Isometry: \n %s" % T_eye)
    array = np.random.rand(3)
    np.testing.assert_allclose(array, T_eye * array, err_msg="*-operator failed")
    np.testing.assert_allclose(array, T_eye @ array, err_msg="@-operator failed")
    np.testing.assert_allclose(
        array, T_eye.transform(array), err_msg="transform-method failed"
    )

    np.testing.assert_allclose(
        trafo.matrix(), (T_eye * trafo).matrix(), err_msg="*-operator failed"
    )
    np.testing.assert_allclose(
        trafo.matrix(), (T_eye @ trafo).matrix(), err_msg="@-operator failed"
    )
    np.testing.assert_allclose(
        trafo.matrix(),
        (T_eye.transform(trafo)).matrix(),
        err_msg="transform-method failed",
    )
    print("Isometry operations: passed")

    np.testing.assert_allclose(np.eye(4), (~t1 @ t1).matrix())
    np.testing.assert_allclose(np.eye(4), (t1 @ ~t1).matrix())
    print("Identity operations: passed")
    """
    t1 = inv1.inverse()
    t2 = inv1
    print(isclose(inv1.rotation, inv2.rotation))
    print(inv1.matrix() == inv2.matrix())
    print(inv1.rotation)
    print(inv2.rotation)
    print(inv1.translation)
    print(inv2.translation)

    t2.invert()

    #print(isclose(t1.matrix(), t2.matrix()))
    #print(isclose(t1.translation, t2.translation))
    print(isclose(t1.rotation, t2.rotation))
    print(t1.rotation)
    print(t2.rotation)
    print(t1.translation)
    print(t2.translation)
    """
