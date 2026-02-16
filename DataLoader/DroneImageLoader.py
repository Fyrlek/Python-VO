import cv2
import numpy as np
import glob
from tqdm import tqdm
import logging

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

        self.cam = PinholeCamera(3840.0, 2160.0, 2620, 1975, 1920, 1080)

        # read ground truth pose
        self.pose_path = self.config["root_path"] + "/poses/" + self.config["sequence"] + ".txt"
        self.gt_poses = []
        with open(self.pose_path) as f:
            lines = f.readlines()
            for line in lines:
                ss = line.strip().split()
                pose = np.zeros((1, len(ss)))
                for i in range(len(ss)):
                    pose[0, i] = float(ss[i])

                pose.resize([3, 4])
                self.gt_poses.append(pose)

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
