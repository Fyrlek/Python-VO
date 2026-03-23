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

    def update(self, image, average_scale, start_R, R_bias):
        """
        update a new image to visual odometry, and compute the pose

        :param image: input image
        :param average_scale: the average scale between frames
        :param start_R: the initial rotation matrix
        :return: R and t of current frame
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
            
        # self.cur_R = R_bias.dot(self.cur_R)

        self.kptdescs["ref"] = self.kptdescs["cur"]

        self.index += 1
        return self.cur_R, self.cur_t
    
    def pre_update(self, image, abs_scale, start_R):
        kptdesc = self.detector(image)
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
            if (abs_scale > 0):
                self.cur_t = self.cur_t + abs_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)

        # self.cur_t = gt_pose[:3, 3].reshape(3, 1)
        # self.cur_R = gt_pose[:3, :3]

        self.kptdescs["ref"] = self.kptdescs["cur"]

        self.index += 1
        return self.cur_R, self.cur_t
            
    def find_alignment_rotation(self, p_est, p_gt):
        """
        Find rotation R such that R @ p_est ≈ p_gt
        
        Args:
            p_est: (N, 3) array of estimated positions
            p_gt:  (N, 3) array of ground truth positions
        
        Returns:
            R: (3, 3) rotation matrix
        """
        # Center both trajectories (remove translation)
        p_est_c = p_est - p_est.mean(axis=0)
        p_gt_c  = p_gt  - p_gt.mean(axis=0)

        # Compute cross-covariance matrix
        H = p_est_c.T @ p_gt_c  # (3, 3)

        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)

        # Rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case (det should be +1, not -1)
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
