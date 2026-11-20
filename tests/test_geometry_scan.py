"""Unit tests for scan geometry (no Kit)."""

from __future__ import annotations

import unittest

import numpy as np

from teleassist.geometry_scan import (
    TOP_DOWN_WXYZ,
    adaptive_grasp_orientation,
    pose_from_depth_points,
    unproject_pixels_lab,
)


class TestGeometryScan(unittest.TestCase):
    def test_top_down_block_orientation(self):
        pts = np.array(
            [
                [0.0, 0.0, 0.1],
                [0.02, 0.0, 0.1],
                [0.0, 0.04, 0.1],
                [0.02, 0.04, 0.11],
            ],
            dtype=np.float64,
        )
        quat = adaptive_grasp_orientation(pts, object_center=np.array([0.01, 0.02, 0.1]), grasp_style="top_down")
        self.assertEqual(quat.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=5)

    def test_pose_from_depth_points(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(loc=[0.1, -0.2, 0.15], scale=0.01, size=(40, 3))
        center, top_z, dims = pose_from_depth_points(cloud)
        self.assertEqual(center.shape, (3,))
        self.assertGreater(top_z, center[2])
        self.assertTrue(np.all(dims > 0))

    def test_unproject_lab_optical_axis(self):
        depth = np.full((8, 8), 1.0, dtype=np.float64)
        intrinsic = np.array([[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float64)
        pos = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        # Identity ROS camera: optical +Z world
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        points = unproject_pixels_lab(np.array([[3.5, 3.5]]), depth, intrinsic, pos, quat)
        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 1.5], atol=1e-6)

    def test_empty_points_fallback_quat(self):
        quat = adaptive_grasp_orientation(np.zeros((0, 3)), grasp_style="auto")
        np.testing.assert_allclose(np.abs(quat), np.abs(TOP_DOWN_WXYZ), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
