# Part 2: Evaluation

## Step 1: Save the trajectory
In order to test the quality of the trajectory that we are able to compute, we must be able to compare it with the ground truth trajectory. For this, we need to save the estimated trajectory in a file:
````c++
  std::vector<cv::Mat> trajectory;
````
and store it when computing it inside the loop:
```c++
  for (const auto& frame_path : frames_in_directory) {
    ...
    trajectory.push_back(pose.clone());
    ...
  }

  cv::destroyAllWindows();
  writeTrajectory(frames_in_directory, trajectory);
  return 0;
}
```
The function `writeTrajectory(frames_in_directory, trajectory)` writes to a file named `estimated_trajectory.txt` the computed trajectory in the correct format (similar to the groundtruth text in the dataset):
```c++
static void writeTrajectory(
    const std::vector<std::filesystem::path>& frames_in_directory,
    const std::vector<cv::Mat>& trajectory) {
  std::ofstream trajectory_file("estimated_trajectory.txt");
  trajectory_file << "# estimated trajectory" << std::endl;
  trajectory_file << "# file: 'rgbd_dataset_freiburg1_xyz.bag'" << std::endl;
  trajectory_file << "# timestamp tx ty tz qx qy qz qw" << std::endl;

  for (int idx = 0; idx < frames_in_directory.size() - 1; ++idx) {
    const auto& frame_path = frames_in_directory[idx];
    auto timestamp_str = frame_path.stem().string();
    auto pose = trajectory[idx];
    auto quat = cv::Quat<double>::createFromRotMat(pose(cv::Rect(0, 0, 3, 3)));
    trajectory_file << timestamp_str << " " << pose.at<double>(0, 3) << " "
                    << pose.at<double>(1, 3) << " " << pose.at<double>(2, 3)
                    << " " << quat.x << " " << quat.y << " " << quat.z << " "
                    << quat.w << std::endl;
  }
  trajectory_file.close();
}
```

## Part 2: Compare trajectories
To compare the trajectories, we are going to use code from [raulmur/evaluate_ate_scale](https://github.com/raulmur/evaluate_ate_scale), stored in the `tools/` folder. We just need to execute the `tools/evaluate_ate_scale.py` file:
```bash
python3 evaluate_ate_scale.py --plot trajectory.png ../data/rgbd_dataset_freiburg1_xyz/groundtruth.txt ../estimated_trajectory.txt 
```