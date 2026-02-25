import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging
import re
from pyproj import Transformer

from utils.PinholeCamera import PinholeCamera


class DroneImageLoader(object):
    default_config = {
        "root_path": "../test_imgs",
        "sequence": "00",
        "start": 0
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Drone Dataset config: ")
        logging.info(self.config)

        self.cam = PinholeCamera(3840.0, 2160.0, 2688, 2688, 1920, 1080)

        # read ground truth pose (SRT) -> parse signed floats, convert to local ENU
        self.pose_path = self.config["root_path"] + "/poses/" + self.config["sequence"] + ".SRT"
        lat_list, lon_list, alt_list = [], [], []

        # robust regex for signed decimal numbers
        num_re = r"([-+]?\d+(?:\.\d+)?)"
        lat_re = re.compile(r"latitude:\s*" + num_re)
        lon_re = re.compile(r"longitude:\s*" + num_re)
        alt_re = re.compile(r"abs_alt:\s*" + num_re)

        with open(self.pose_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                mlat = lat_re.search(line)
                mlon = lon_re.search(line)
                malt = alt_re.search(line)
                if mlat and mlon and malt:
                    lat_list.append(float(mlat.group(1)))
                    lon_list.append(float(mlon.group(1)))
                    alt_list.append(float(malt.group(1)))

        assert len(lat_list) > 0, "No GPS data found in SRT file"
        assert len(lat_list) == len(lon_list) == len(alt_list)

        # Convert geodetic -> ECEF
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        ecef = np.array([transformer.transform(lon, lat, alt) for lat, lon, alt in zip(lat_list, lon_list, alt_list)])

        # compute local ENU coordinates relative to first point
        ecef0 = ecef[0]
        lat0 = np.deg2rad(lat_list[0])
        lon0 = np.deg2rad(lon_list[0])
        slat = np.sin(lat0); clat = np.cos(lat0)
        slon = np.sin(lon0); clon = np.cos(lon0)

        # ECEF -> ENU rotation matrix (standard)
        R = np.array([[-slon,           clon,            0.0],
                        [-slat*clon,     -slat*slon,      clat],
                        [ clat*clon,      clat*slon,      slat]])

        deltas = (ecef - ecef0)
        enu = (R @ deltas.T).T   # columns: [E, N, U]

        # populate gt poses using ENU. For 2D trajectory plotting the code expects
        # pose[0] -> x (east) and pose[2] -> z (north), so we place E in index 0 and N in index 2.
        # optionally rotate the gt coordinate frame about the origin using a rotation matrix
        # supplied in the config (e.g. to correct an unknown yaw).
        self.gt_poses = []
        # pre‑compute rotation matrix if provided
        rot_mat = np.eye(3)
        if "gt_rot" in self.config:
            r = np.array(self.config["gt_rot"])  # expect 3x3 list
            assert r.shape == (3, 3), "gt_rot must be a 3x3 matrix"
            rot_mat = r.astype(np.float64)
        elif "gt_yaw_deg" in self.config:
            # yaw angle in degrees. The loader maps ENU -> pose as
            # x=E (index 0), y=vertical (index 1), z=N (index 2),
            # so the up axis is the pose y axis. Rotate about y.
            yaw = np.deg2rad(self.config["gt_yaw_deg"])
            c, s = np.cos(yaw), np.sin(yaw)
            # rotation about Y (up) axis
            rot_mat = np.array([[ c, 0.0,  s],
                                [0.0,1.0, 0.0],
                                [-s, 0.0,  c]])

        for e, n, u in enu:
            pose = np.zeros((3, 4), dtype=np.float64)
            pose[:3, :3] = rot_mat  # apply optional rotation
            pose[0, 3] = e   # x = East
            pose[1, 3] = -u # keep y zero (or could use -u if desired)
            pose[2, 3] = n   # z = North
            # rotate translation as well (apply rotation to the vector)
            pose[:3, 3] = rot_mat.dot(pose[:3, 3])
            self.gt_poses.append(pose)
        
        #write gt poses to txt file
        with open(self.config["root_path"] + "/poses/" + self.config["sequence"] + "_gt.txt", "w", encoding="utf-8") as f:
            for pose in self.gt_poses:
                pose_str = " ".join([str(x) for x in pose.flatten()])
                f.write(pose_str + "\n")

        # image id
        self.img_id = self.config["start"]
        self.img_N = len(glob.glob(pathname=self.config["root_path"] + "/sequences/" + self.config["sequence"] + "/*.jpg"))

    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]

    def __getitem__(self, item):
        file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] + "/" + str(item).zfill(6) + ".jpg"
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] + "/" + str(self.img_id).zfill(6) + ".jpg"
            img = cv2.imread(file_name)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]


if __name__ == "__main__":
    loader = DroneImageLoader()

    for img in tqdm(loader):
        cv2.putText(img, "Press any key but Esc to continue, press Esc to exit", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)
        cv2.imshow("img", img)
        # press Esc to exit
        if cv2.waitKey() == 27:
            break
