import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import json 
import random

JOINTS = ['r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
          'pelvis', 'thorax', 'upper_neck', 'head_top', 'r_wrist', 'r_elbow',
          'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

ANNOTATION_FOLDER = "annotation"
IMAGE_FOLDER = "img"

class ComposedDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        print("Loader started.")
        self.data_dir = os.path.abspath(data_dir)
        self.P_ID = list()
        self.joints = self.get_joints_rgb(data_dir)
        self.size = len(self.joints)

        print(f"{self.size} images loaded.\n")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        name = list(self.joints.keys())[idx]
        img = cv2.imread(os.path.join(self.data_dir, IMAGE_FOLDER, name))
        img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

        kpts = [i for i in self.joints[name].values()]
        kpts = np.array(kpts)
        kpts = kpts[np.newaxis, :]

        return img, kpts, name

    def get_joints_rgb(self, data_path: str):
        """
        :param
            data_path : str
                path to watch-n-patch .jpg file
        :return
            joints : dict
                dictionary with frame_path as key and joints annotation as value
        """
        image_name_to_joints = dict()
        image_path = os.path.join(self.data_dir, IMAGE_FOLDER)
        annotation_path = os.path.join(self.data_dir, ANNOTATION_FOLDER)
        generic_joint_json = {joint : (random.randint(10,50), random.randint(10,50)) for joint in JOINTS}

        print(image_path)

        image_names = self.get_image_names(image_path)
        for image_name in image_names:
            key = image_name.split('.')[0]

            if key + ".json" not in os.listdir(annotation_path):
                generic_joint_json_object = json.dumps(generic_joint_json, indent=4)
                with open(os.path.join(annotation_path, key + ".json"), "w") as outfile:
                    outfile.write(generic_joint_json_object)
            
            with open(os.path.join(annotation_path, key + ".json")) as json_file:
                image_name_to_joints[image_name] = json.load(json_file)
            
        return image_name_to_joints

    def get_image_names(self, img_dir: str):
        images = os.listdir(img_dir)
        images.sort()
        if any(".DS_Store" in s for s in images):
            images.remove(".DS_Store")
        if any("._.DS_Store" in s for s in images):
            images.remove("._.DS_Store")
        return images
    