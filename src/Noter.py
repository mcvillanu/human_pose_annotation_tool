from RGB_Dataset import ComposedDataset as RGB_Dataset
import numpy as np
from RGB_Dataset import JOINTS, ANNOTATION_FOLDER
import json
import os
import cv2
import tkinter
import tkinter.ttk as ttk
from tkinter import simpledialog
from torch.utils.data import Dataset
import argparse

# Split we want to note down
SPLIT = ['data_03-58-25', 'data_03-25-32', 'data_02-32-08', 'data_03-05-15', 'data_11-11-59',
         'data_03-21-23', 'data_03-35-07', 'data_03-04-16', 'data_04-30-36', 'data_02-50-20',
         'data_04-51-42', 'data_04-52-02', 'data_02-10-35', 'data_03-45-21', 'data_03-53-06',
         'data_12-07-43', 'data_05-04-12', 'data_04-27-09', 'data_04-13-06', 'data_01-52-55']

# Color selection for visualization
JOINTS_COLOR = [(255, 0, 0), (244, 41, 0), (234, 78, 0), (223, 112, 0),
                (213, 142, 0), (202, 168, 0), (192, 191, 0), (151, 181, 0),
                (213, 142, 0), (202, 168, 0), (192, 191, 0), (151, 181, 0),
                (114, 170, 0), (80, 160, 0), (50, 149, 0), (23, 139, 0),
                (114, 170, 0), (80, 160, 0), (50, 149, 0), (23, 139, 0),
                (244, 41, 0)]

class Noter:
    """
        Main class for the Noter tool

    """
    def __init__(self, d: Dataset, ann_path, scale, radius, next_skip):
        # PyTorch Dataset, you can define your own version for custom use
        # next_skip : int -> how many frame you skip
        # tot : int -> how many split we want to note down
        self.dataset = d
        self.next_skip = next_skip

        # Path where you can load and save annotations
        self.annotation_path = ann_path

        # Visualization variable, radius for circle dimension and scale for image dimension
        self.radius = radius
        self.scale = scale

        # Loading annotation if we're resuming old noting sessions
        self.json_dict = dict()
        if os.path.isfile(self.annotation_path):
            with open(self.annotation_path, 'r') as f:
                self.json_dict = json.load(f)

        # Event variable, for click, modify and add events
        self.is_clicked = False
        self.is_modifying = False
        self.is_adding_joint = False

        # Point position and index variable
        self.point = [-1, -1]
        self.kpt_idx = -1
        self.obj_idx = -1

        # Tkinter attribute for windows management
        self.master = tkinter.Tk()
        self.info = tkinter.StringVar()
        self.error = tkinter.StringVar()
        self.status = tkinter.StringVar()
        self.sequences = tkinter.StringVar()
        self.progressbar = ttk.Progressbar(self.master, orient="horizontal", length=100, mode="determinate")
        self.progressbar.pack(side=tkinter.BOTTOM)
        tkinter.Label(master=self.master, textvariable=self.info).pack()
        tkinter.Label(master=self.master, textvariable=self.error, width=25).pack()
        tkinter.Label(master=self.master, textvariable=self.status).pack()
        tkinter.Label(master=self.master, textvariable=self.sequences).pack()

        # Tkinter initialization
        self.info.set("....")
        self.error.set("....")
        self.status.set("....")
        self.sequences.set("....")

        # Select format for different system
        if os.name == 'nt':
            self.slash = '\\'
        elif os.name == 'posix':
            self.slash = '/'
        else:
            raise NotImplementedError(f"Wrong system, implement different path management: {os.name}")

    def start(self):
        # Starting function
        self.master.update()
        next_name = None
        for i, (imgs, kpts, names) in enumerate(self.dataset):
            print(names)
            key = names.split('.')[0]
            current_json_path = os.path.join(self.dataset.data_dir, ANNOTATION_FOLDER, key + ".json")

            with open(current_json_path, 'r') as f:
                self.current_json_dict = json.load(f)

            if isinstance(names, list):
                name = names[0]
            else:
                name = names

            new_imgs = list()
            if isinstance(imgs, list):
                img = imgs[0]
                for index, new_img in enumerate(imgs[1:]):
                    new_imgs.append(cv2.resize(new_img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
                    cv2.namedWindow(names[index + 1])
                    cv2.moveWindow(names[index + 1], 750, 300)
            else:
                img = imgs

            tmp = img.astype(np.uint8).copy()
            kpts_back = kpts.copy()

            self.draw_kpts(tmp, kpts, self.radius)
            cv2.namedWindow(name)
            cv2.moveWindow(name, 15, 150)
            cv2.setMouseCallback(name, self.click_left, [name, tmp, kpts])

            while True:
                if len(new_imgs) > 0:
                    for new_name, el in zip(names[1:], new_imgs):
                        cv2.imshow(new_name, el)
                cv2.imshow(name, tmp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('\n') or key == ord('\r'):
                    self.current_json_dict = dict(zip(JOINTS, kpts.copy().tolist()[0]))
                    cv2.destroyAllWindows()
                    with open(current_json_path, 'w') as f:
                        json.dump(self.current_json_dict, f)
                    self.reset()
                    self.info.set("....")
                    self.error.set("....")
                    self.master.update()
                    break

                elif key == ord('r'):
                    tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                    np.copyto(kpts, kpts_back)
                    self.draw_kpts(tmp, kpts, self.radius)
                    self.reset()

                elif key == ord('c'):
                    with open(current_json_path, 'w') as f:
                        json.dump(self.current_json_dict, f)
                    exit(1)

                elif key == ord('p'):
                    self.error.set("Changing sequence.")
                    self.master.update()
                    next_name = name.split(self.slash)[-3]
                    self.reset()
                    break

                elif key == ord('n'):
                    tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                    np.copyto(kpts, kpts_back)
                    self.draw_kpts(tmp, kpts, self.radius)
                    self.reset()

                if self.is_modifying is not True:
                    if key == 27 and self.is_clicked is True:
                        for obj in range(kpts.shape[0]):
                            for el in kpts[obj]:
                                if int(el[0]) == self.point[0] and int(el[1]) == self.point[1]:
                                    el[0] = -1
                                    el[1] = -1
                                    break
                        tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                        self.draw_kpts(tmp, kpts, self.radius)
                        self.is_clicked = False
                        self.is_modifying = True

                    elif key == ord('a') and self.is_clicked is not True:
                        obj = 0
                        val = 0
                        lgnd = ""
                        self.info.set('....')
                        self.error.set('Adding joint.')
                        self.master.update()
                        while self.kpt_idx < 0 and val is not None and obj is not None:
                            for key, v in enumerate(JOINTS):
                                if key > 20:
                                    break
                                lgnd += "{} for {}".format(key, v)
                                if key != 20:
                                    lgnd += ", "
                                    if (key + 1) % 3 == 0:
                                        lgnd += '\n'
                            val = simpledialog.askinteger("Input", "Insert Joint Number\n{}".format(lgnd),
                                                          parent=self.master,
                                                          minvalue=0, maxvalue=25)
                            self.master.update()
                            if val:
                                if kpts.shape[0] > 1:
                                    obj = simpledialog.askinteger("Input", "Insert obj number",
                                                                  parent=self.master,
                                                                  minvalue=0, maxvalue=kpts.shape[0])
                                if obj is not None:
                                    self.kpt_idx = val
                                    self.obj_idx = obj
                                    if self.kpt_idx < 0 or self.kpt_idx > 20:
                                        self.error.set("Insert only value between 0 and 20.")
                                        self.master.update()
                                        self.kpt_idx = -1
                                        continue
                                    self.is_adding_joint = True
                else:
                    if key == ord('y') and self.is_modifying is True:
                        self.error.set('....')
                        tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                        np.copyto(kpts_back, kpts)
                        print('tmp')
                        print(tmp[:, :, :])
                        print('kpts back')
                        print(kpts_back)
                        print('kpts ')

                        print(kpts)
                        self.draw_kpts(tmp, kpts, self.radius)
                        self.reset()

        with open(current_json_path, 'w') as f:
            json.dump(self.current_json_dict, f)
        exit(1)

    def reset(self):
        self.is_modifying = False
        self.is_clicked = False
        self.is_adding_joint = False
        self.obj_idx = -1
        self.kpt_idx = -1
        self.point = [-1, -1]

    @staticmethod
    def draw_kpts(img, kpts, radius):
        print('kpts INSIE DRAW')
        print(kpts)
        for obj in range(kpts.shape[0]):
            for i, el in enumerate(kpts[obj]):
                if i > 20:
                    break
                if el[0] >= 0 and el[1] >= 0:
                    cv2.circle(img, (int(el[0]), int(el[1])), radius, JOINTS_COLOR[i], -1)
                    cv2.putText(img,'{}'.format(JOINTS[i]), (int(el[0]) + 5, int(el[1])+ 5), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,0), thickness=2)
    def search_near(self, x, y, kpts):
        for obj in range(kpts.shape[0]):
            for i, el in enumerate(kpts[obj]):
                if i > 20:
                    break
                range_x = [el[0] - self.radius + 2, el[0] + self.radius + 2]
                range_y = [el[1] - self.radius + 2, el[1] + self.radius + 2]
                if range_x[0] <= x <= range_x[1] and range_y[0] <= y <= range_y[1]:
                    self.info.set(JOINTS[i])
                    self.master.update()
                    return int(el[0]), int(el[1])
        return -1, -1

    def click_left(self, event, x, y, flags, param):
        name = param[0]
        img = param[1]
        kpts = param[2]
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_modifying is False:
                if self.is_clicked is True:
                    self.is_clicked = False
                    self.is_modifying = True
                    old_x, old_y = self.point
                    self.point = [-1, -1]
                    for obj in range(kpts.shape[0]):
                        for el in kpts[obj]:
                            if int(el[0]) == old_x and int(el[1]) == old_y:
                                el[0] = x
                                el[1] = y
                                break
                    cv2.circle(img, (x, y), self.radius, (0, 0, 255), -1)
                    cv2.imshow(name, img)
                elif self.is_adding_joint is True:
                    kpts[self.obj_idx][self.kpt_idx][0] = x
                    kpts[self.obj_idx][self.kpt_idx][1] = y
                    cv2.circle(img, (x, y), self.radius, (0, 255, 255), -1)
                    cv2.imshow(name, img)
                    self.error.set("{} added.".format(JOINTS[self.kpt_idx]))
                    self.master.update()
                    self.is_modifying = True
                    self.is_adding_joint = False
                    self.kpt_idx = -1
                    self.obj_idx = -1
                else:
                    if tuple(img[y, x]) in JOINTS_COLOR:
                        new_x, new_y = self.search_near(x, y, kpts)
                        if new_y > 0 and new_x > 0:
                            cv2.circle(img, (new_x, new_y), self.radius, (0, 0, 255), -1)
                            self.is_clicked = True
                            self.point = [new_x, new_y]
                            cv2.imshow(name, img)
            else:
                self.error.set("Confirm modfying?")
                self.master.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, dest='data_dir', help='Data directory.', required=True)
    parser.add_argument('--out', type=str, default='good_annotations', dest='out', help='Output file path.')
    parser.add_argument('--scale', type=float, default=1, dest='scale', help='Depth image scale.')
    parser.add_argument('--radius', type=int, default=6, dest='radius', help='Joint annotation radius.')
    parser.add_argument('--next', type=int, default=1, dest='next', help='Skippin [next] image.')
    args = parser.parse_args().__dict__

    n = Noter(RGB_Dataset(args['data_dir']), args['out'], args['scale'], args['radius'], args['next'])
    n.start()
