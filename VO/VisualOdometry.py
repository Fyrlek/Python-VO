# based on: https://github.com/uoip/monoVO-python

import numpy as np
import cv2


class VisualOdometry(object):
    """
    A simple frame by frame visual odometry
    """

    def __init__(self, detector, matcher, cam):
        """
        :param detector: a feature detector can detect keypoints their descriptors
        :param matcher: a keypoints matcher matching keypoints between two frames
        :param cam: camera parameters
        """
        # feature detector and keypoints matcher
        self.detector = detector
        self.matcher = matcher

        # camera parameters
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        # frame index counter
        self.index = 0

        # keypoints and descriptors
        self.kptdescs = {}

        # pose of current frame
        self.cur_R = None
        self.cur_t = None

        # accumulated positions during pre_update phase for alignment
        self.pre_est_positions = []
        self.pre_gt_positions = []

    def update(self, image, average_scale, start_R, R_bias=None):
        """
        update a new image to visual odometry, and compute the pose

        :param image: input image
        :param average_scale: the average scale between frames
        :param start_R: the initial rotation matrix
        :param R_bias: optional (3,3) alignment rotation applied to output only, not internal state
        :return: R and t of current frame (with R_bias applied to t for display)
        """
        kptdesc = self.detector(image)
        # update keypoints and descriptors
        self.kptdescs["cur"] = kptdesc

        # first frame
        if self.index == 0:
            # start point
            self.cur_R = np.array(start_R)
            self.cur_t = np.zeros((3, 1))
        else:
            # match keypoints
            matches = self.matcher(self.kptdescs)

            # compute relative R,t between ref and cur frame
            E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],
                                           focal=self.focal, pp=self.pp,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                            focal=self.focal, pp=self.pp)

            # get absolute pose based on absolute_scale
            if (average_scale > 0):
                self.cur_t = self.cur_t + average_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)

        self.kptdescs["ref"] = self.kptdescs["cur"]
        self.index += 1

        # Apply R_bias to output only — never mutate internal state
        if R_bias is not None:
            return R_bias @ self.cur_R, R_bias @ self.cur_t
        return self.cur_R, self.cur_t
    
    def pre_update(self, image, abs_scale, start_R, gt_pose):
        kptdesc = self.detector(image)
        self.kptdescs["cur"] = kptdesc
        # first frame
        if self.index == 0:
            self.cur_R = np.array(start_R)
            self.cur_t = np.zeros((3, 1))
        else:
            # match keypoints
            matches = self.matcher(self.kptdescs)

            # compute relative R,t between ref and cur frame
            E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],
                                           focal=self.focal, pp=self.pp,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                            focal=self.focal, pp=self.pp)

            if (abs_scale > 0):
                self.cur_t = self.cur_t + abs_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)

        # accumulate for Procrustes alignment
        self.pre_est_positions.append(self.cur_t.flatten().copy())
        self.pre_gt_positions.append(gt_pose[:3, 3].copy())

        self.kptdescs["ref"] = self.kptdescs["cur"]
        self.index += 1
        return self.cur_R, self.cur_t

    def compute_alignment_rotation(self):
        """
        Solves the Orthogonal Procrustes problem to find R_bias such that R_bias @ p_est ≈ p_gt.
        R_bias is meant to be passed to update() for output-only correction.
        """
        p_est = np.array(self.pre_est_positions)  # (N, 3)
        p_gt  = np.array(self.pre_gt_positions)   # (N, 3)
        return self.find_alignment_rotation(p_est, p_gt)

    def find_alignment_rotation(self, p_est, p_gt):
        """
        Find rotation R such that R @ p_est ≈ p_gt  (Orthogonal Procrustes)
        """
        p_est_c = p_est - p_est.mean(axis=0)
        p_gt_c  = p_gt  - p_gt.mean(axis=0)

        H = p_est_c.T @ p_gt_c

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection (det must be +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return R
            

class AbosluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose

        scale = 1.0
        if self.count != 0:
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) * (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) * (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) * (self.cur_pose[2, 3] - self.prev_pose[2, 3]))

        self.count += 1
        self.prev_pose = self.cur_pose
        return scale


if __name__ == "__main__":
    from DataLoader.KITTILoader import KITTILoader
    from Detectors.HandcraftDetector import HandcraftDetector
    from Matchers.FrameByFrameMatcher import FrameByFrameMatcher

    loader = KITTILoader()
    detector = HandcraftDetector({"type": "SIFT"})
    matcher = FrameByFrameMatcher({"type": "FLANN"})
    absscale = AbosluteScaleComputer()

    vo = VisualOdometry(detector, matcher, loader.cam)
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()
        R, t = vo.update(img, absscale.update(gt_pose))
