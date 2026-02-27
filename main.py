import os

import numpy as np
import cv2
import argparse
import yaml
import logging
import open3d as o3d

from utils.tools import plot_keypoints
from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer


def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"])


class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        # store 3D points
        self.est_points = []
        self.gt_points = []
        
        # Open3D visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="3D Trajectory", width=600, height=600)
        # state for orthographic toggle and saved zoom
        self._is_orthographic = False
        self._orthographic_sim = False
        self._saved_zoom = None

        # Register callbacks
        self.vis.register_key_callback(ord("1"), lambda vis: self._snap_axis("x"))
        self.vis.register_key_callback(ord("2"), lambda vis: self._snap_axis("y"))
        self.vis.register_key_callback(ord("3"), lambda vis: self._snap_axis("z"))
        
        # line sets for trajectories
        self.est_lineset = None
        self.gt_lineset = None
        
        # point clouds for endpoints
        self.est_pcd = o3d.geometry.PointCloud()
        self.gt_pcd = o3d.geometry.PointCloud()

    def _snap_axis(self, axis):
        vc = self.vis.get_view_control()
        # compute scene center
        if len(self.est_points) + len(self.gt_points) > 0:
            pts = np.vstack(self.est_points + self.gt_points)
            center = pts.mean(axis=0)
        else:
            center = np.array([0.0, 0.0, 0.0])

        if axis == "x":
            front = [1, 0, 0]
            up = [0, 0, 1]
        elif axis == "y":
            front = [0, 1, 0]
            up = [0, 0, 1]
        else:
            front = [0, 0, 1]
            up = [0, 1, 0]

        vc.set_front(front)
        vc.set_up(up)
        vc.set_lookat(center)
        vc.set_zoom(0.5)

        return False

    def update(self, est_xyz, gt_xyz):
        # convert inputs to flat 3D vectors
        est = np.asarray(est_xyz).reshape(3,)
        gt = np.asarray(gt_xyz).reshape(3,)

        # append points
        self.est_points.append(est)
        self.gt_points.append(gt)

        # compute 3D error
        err = float(np.linalg.norm(est - gt))
        self.errors.append(err)
        avg_error = float(np.mean(np.array(self.errors)))

        # clear previous geometries
        self.vis.clear_geometries()

        # build estimated trajectory line
        est_array = np.array(self.est_points)
        if est_array.shape[0] > 1:
            lines_est = [[i, i + 1] for i in range(est_array.shape[0] - 1)]
            self.est_lineset = o3d.geometry.LineSet()
            self.est_lineset.points = o3d.utility.Vector3dVector(est_array)
            self.est_lineset.lines = o3d.utility.Vector2iVector(lines_est)
            self.est_lineset.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines_est])  # green
            self.vis.add_geometry(self.est_lineset)
        elif est_array.shape[0] > 0:
            self.est_pcd.points = o3d.utility.Vector3dVector(est_array)
            self.est_pcd.paint_uniform_color([0, 1, 0])
            self.vis.add_geometry(self.est_pcd)

        # build ground-truth trajectory line
        gt_array = np.array(self.gt_points)
        if gt_array.shape[0] > 1:
            lines_gt = [[i, i + 1] for i in range(gt_array.shape[0] - 1)]
            self.gt_lineset = o3d.geometry.LineSet()
            self.gt_lineset.points = o3d.utility.Vector3dVector(gt_array)
            self.gt_lineset.lines = o3d.utility.Vector2iVector(lines_gt)
            self.gt_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines_gt])  # red
            self.vis.add_geometry(self.gt_lineset)
        elif gt_array.shape[0] > 0:
            self.gt_pcd.points = o3d.utility.Vector3dVector(gt_array)
            self.gt_pcd.paint_uniform_color([1, 0, 0])
            self.vis.add_geometry(self.gt_pcd)

        # Add coordinate frame at origin; scale according to scene extents
        if len(self.est_points) + len(self.gt_points) > 0:
            pts = np.vstack(self.est_points + self.gt_points)
            span = np.max(pts, axis=0) - np.min(pts, axis=0)
            size = float(np.linalg.norm(span)) * 0.05  # axis length % of diagonal
            size = max(size, 1.0)  # minimum
        else:
            size = 1.0
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        self.vis.add_geometry(mesh)

        # Update view
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # Capture screen as image for cv2.imshow
        img = self.vis.capture_screen_float_buffer(do_render=True)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Add background rectangle for text
        (text_w, text_h), _ = cv2.getTextSize(f"AvgError: {avg_error:.4f} m", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_bgr, (5, 5), (15 + text_w, 25 + text_h), (0, 0, 0), -1)
        
        # Add error text overlay
        cv2.putText(img_bgr, f"AvgError: {avg_error:.4f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img_bgr

    def finalize(self):
        """Keep the visualizer window open for interactive inspection.
        Call this once after all updates are done (e.g. at end of run())."""
        # start interaction loop - this blocks until the window is closed
        self.vis.run()
        self.vis.destroy_window()


def run(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # create dataloader
    loader = create_dataloader(config["dataset"])
    # create detector
    detector = create_detector(config["detector"])
    # create matcher
    matcher = create_matcher(config["matcher"])

    absscale = AbosluteScaleComputer()
    traj_plotter = TrajPlotter()

    # log
    fname = args.config.split('/')[-1].split('.')[0]
    log_fopen = open("results/" + fname + ".txt", mode='a')

    vo = VisualOdometry(detector, matcher, loader.cam)
    for i, img in enumerate(loader):
        if i % 10 == 0:
            gt_pose = loader.get_cur_pose()
            R, t = vo.update(img, absscale.update(gt_pose))
            # R, t = vo.update(img, 1.0)

            # === log writer ==============================
            print(i, t[0, 0], t[1, 0], t[2, 0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3], file=log_fopen)

            # === drawer ==================================
            img1 = keypoints_plot(img, vo)
            img2 = traj_plotter.update(t, gt_pose[:, 3])

            cv2.imshow("keypoints", img1)
            cv2.imshow("trajectory", img2)
            if cv2.waitKey(10) == 27:
                break

    cv2.imwrite("results/" + fname + '.png', img2)

    # keep 3D plot interactive after processing
    traj_plotter.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
    parser.add_argument('--config', type=str, default='params/kitti_superpoint_supergluematch.yaml',
                        help='config file')
    parser.add_argument('--logging', type=str, default='INFO',
                        help='logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL')

    args = parser.parse_args()

    logging.basicConfig(level=logging._nameToLevel[args.logging])

    run(args)
