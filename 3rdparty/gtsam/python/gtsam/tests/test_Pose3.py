"""
GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

Pose3 unit tests.
Author: Frank Dellaert, Duy Nguyen Ta
"""
# pylint: disable=no-name-in-module
import math
import unittest

import numpy as np

import gtsam
from gtsam import Point3, Pose3, Rot3, Point3Pairs
from gtsam.utils.test_case import GtsamTestCase


class TestPose3(GtsamTestCase):
    """Test selected Pose3 methods."""

    def test_between(self):
        """Test between method."""
        T2 = Pose3(Rot3.Rodrigues(0.3, 0.2, 0.1), Point3(3.5, -8.2, 4.2))
        T3 = Pose3(Rot3.Rodrigues(-90, 0, 0), Point3(1, 2, 3))
        expected = T2.inverse().compose(T3)
        actual = T2.between(T3)
        self.gtsamAssertEquals(actual, expected, 1e-6)

    def test_transformTo(self):
        """Test transformTo method."""
        pose = Pose3(Rot3.Rodrigues(0, 0, -math.pi/2), Point3(2, 4, 0))
        actual = pose.transformTo(Point3(3, 2, 10))
        expected = Point3(2, 1, 10)
        self.gtsamAssertEquals(actual, expected, 1e-6)

        # multi-point version
        points = np.stack([Point3(3, 2, 10), Point3(3, 2, 10)]).T
        actual_array = pose.transformTo(points)
        self.assertEqual(actual_array.shape, (3, 2))
        expected_array = np.stack([expected, expected]).T
        np.testing.assert_allclose(actual_array, expected_array, atol=1e-6)

    def test_transformFrom(self):
        """Test transformFrom method."""
        pose = Pose3(Rot3.Rodrigues(0, 0, -math.pi/2), Point3(2, 4, 0))
        actual = pose.transformFrom(Point3(2, 1, 10))
        expected = Point3(3, 2, 10)
        self.gtsamAssertEquals(actual, expected, 1e-6)

        # multi-point version
        points = np.stack([Point3(2, 1, 10), Point3(2, 1, 10)]).T
        actual_array = pose.transformFrom(points)
        self.assertEqual(actual_array.shape, (3, 2))
        expected_array = np.stack([expected, expected]).T
        np.testing.assert_allclose(actual_array, expected_array, atol=1e-6)

    def test_range(self):
        """Test range method."""
        l1 = Point3(1, 0, 0)
        l2 = Point3(1, 1, 0)
        x1 = Pose3()

        xl1 = Pose3(Rot3.Ypr(0.0, 0.0, 0.0), Point3(1, 0, 0))
        xl2 = Pose3(Rot3.Ypr(0.0, 1.0, 0.0), Point3(1, 1, 0))

        # establish range is indeed zero
        self.assertEqual(1, x1.range(point=l1))

        # establish range is indeed sqrt2
        self.assertEqual(math.sqrt(2.0), x1.range(point=l2))

        # establish range is indeed zero
        self.assertEqual(1, x1.range(pose=xl1))

        # establish range is indeed sqrt2
        self.assertEqual(math.sqrt(2.0), x1.range(pose=xl2))

    def test_adjoint(self):
        """Test adjoint methods."""
        T = Pose3()
        xi = np.array([1, 2, 3, 4, 5, 6])
        # test calling functions
        T.AdjointMap()
        T.Adjoint(xi)
        T.AdjointTranspose(xi)
        Pose3.adjointMap(xi)
        Pose3.adjoint(xi, xi)
        # test correctness of adjoint(x, y)
        expected = np.dot(Pose3.adjointMap_(xi), xi)
        actual = Pose3.adjoint_(xi, xi)
        np.testing.assert_array_equal(actual, expected)

    def test_serialization(self):
        """Test if serialization is working normally"""
        expected = Pose3(Rot3.Ypr(0.0, 1.0, 0.0), Point3(1, 1, 0))
        actual = Pose3()
        serialized = expected.serialize()
        actual.deserialize(serialized)
        self.gtsamAssertEquals(expected, actual, 1e-10)

    def test_align_squares(self):
        """Test if Align method can align 2 squares."""
        square = np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]], float).T
        sTt = Pose3(Rot3.Rodrigues(0, 0, -math.pi), Point3(2, 4, 0))
        transformed = sTt.transformTo(square)

        st_pairs = Point3Pairs()
        for j in range(4):
            st_pairs.append((square[:,j], transformed[:,j]))

        # Recover the transformation sTt
        estimated_sTt = Pose3.Align(st_pairs)
        self.gtsamAssertEquals(estimated_sTt, sTt, 1e-10)

        # Matrix version
        estimated_sTt = Pose3.Align(square, transformed)
        self.gtsamAssertEquals(estimated_sTt, sTt, 1e-10)


if __name__ == "__main__":
    unittest.main()
