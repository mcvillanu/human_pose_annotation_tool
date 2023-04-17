<h1 align="center">Human Pose Annotation Tool</h1>

![Example image_1](https://github.com/aimagelab/human-pose-annotation-tool/blob/master/img/example_1.png)

We aim to create a simple yet effective tool for create and modify annotation for Body Pose Estimation over RGB images.

Our tool can be used for annotating new images, or adjusting existing images. The annotation outputs are in a simple to use JSON format. Joints that are not included in the image are denoted as [-1,-1].

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. There's not much to do, just install prerequisites and download all the files.

### Prerequisites
Things you need to install to run the tool:

```
Python >= 3.6.7
pip install numpy
pip install opencv-python
pip install scipy
```

For running our test and to use our code without modifying a line, install PyTorch to use Dataset class.
```
pip install pytorch
```

## Running the tests
To run the simple test, unzip the archive inside the test folder, then execute:
```
python src/Noter.py --data_dir ./test
```
This will open up one sequence of the [Watch-N-Patch](http://watchnpatch.cs.cornell.edu/ "WnP Page") dataset, in which the 2 frames shown above came from. The sequence is stored in the test directory.

## Features
- **Functionality**, shown in the tkinter menu.
  - **Move Joints**, click the joint you want to move, click the new position, press "y" to confirm.
  - **Delete Joints**, click the joint, press "esc" and confirm with "y".
  - **Add not noted Joints**. press "a", choose the joint you want to add, choose position and press "y" to confirm.
  - **Reset action**, after an action, press "esc" to reset.
  - **Save frame**, press "enter" to save and change frame.
  - **Skip sequence**, press "p" to skip the entire sequence.
- **Input**, as default we used a PyTorch dataset wich return depth maps and keypoints value of the Watch-N-Patch Dataset. 
For custom use you need to redifine the Dataset class and load the correct RGB and Depth Images.
With little code modification you can change the tool to work with only RGB or only Depth frames.
- **Output**, JSON file containing image path as key and keypoints array as value.
- **OS**, tested on Windows and Linux OS.

## Usage
```
python Noter.py --data_dir <path_to_dataset> --out <optional> --k <optional> --next <optional>  
                --scale <optional> --radius <optional> --split <optional>
```
- `--out`, putput name file. Default: "good_annotations".
- `--k`, choose if you want to resume annotations from the file specified in `--k` flag, enter "skip" for resume or "keep" for restarting from frame 1.
- `--next`, choose how many frame you want to skip after everyone you note down.
- `--scale` & -`--radius`, choose the dimension of images and keypoint visualization in the plot.
- `--split`, additional parameter for custom datasets.