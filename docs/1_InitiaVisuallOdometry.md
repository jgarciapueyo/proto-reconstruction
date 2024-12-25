# Part 1: Initial Visual Odometry
A basic visual odometry program consists on estimating important features on two consecutive frames, matching these features and, then, extract the essential matrix from the matches. From the essential matrix, and taking into account the calibration matrix, we are able to recover the pose (translation and rotation) between these two frames. By concatenating the poses of every two consecutive frames, we can compute the trajectory of the camera.

However, with monocular visual odometry, the scale of the translation of the pose is up to scale, so we can not concatenate two arbitrary poses due to the scale ambiguity problem. There are different approaches like using known objects in the scene to derive the scale, use SfM with bundle adjustment between frames or, the simplest version that we are goint to use initially, assume constant velocity and, thus, normalize the translation between every pair of frames to assume its norm is one.

## Step 1: Download the 'freiburg1_xyz' dataset
We are goint to use a basic dataset, the [freiburg1_xyz](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg1_xyz) dataset, which consists of a typical desk in an office environment with translation movements along the principal axes (x,y,z) with the orientation mostly fixed.

1. Create a folder to store the dataset:
```bash
mkdir data
```
2. Download the dataset:
```bash
curl curl -o data/freiburg1_xyz.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
```
3. Extract the dataset:
```bash
tar -xvzf data/freiburg1_xyz.tgz -C data/
```

The dataset contains depth and rgb images (in `depth/` and `rgb/` folders), and data about the accelerometer and a groundtruth trajectory defining the translation and rotation.

## Step 2: Read the dataset and show it
````c++
#include <filesystem>
#include <opencv2/opencv.hpp>

int main() {
  // 1. Read path of frames
  const std::string path{"../data/rgbd_dataset_freiburg1_xyz/rgb"};
  std::vector<std::filesystem::path> frames_in_directory;
  std::copy(std::filesystem::directory_iterator(path),
            std::filesystem::directory_iterator(),
            std::back_inserter(frames_in_directory));
  std::sort(frames_in_directory.begin(), frames_in_directory.end());

  // 2. Display the images
  for (const auto& frame_path : frames_in_directory) {
    auto current_frame = cv::imread(frame_path.string());
    cv::imshow("Display Frame", current_frame);
    cv::waitKey(0);
  }

  cv::destroyAllWindows();
  return 0;
}
````

## Step 3: Simple Visual Odometry