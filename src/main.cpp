#include <filesystem>
#include <fstream>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/opencv.hpp>

#include "proto_recon/visualization/visualizer.h"
#include "proto_recon/vo/frame.h"

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

int main() {
  // 1. Read path of images
  const std::string path{"../data/rgbd_dataset_freiburg1_xyz/rgb"};
  std::vector<std::filesystem::path> imgs_in_directory;
  std::copy(std::filesystem::directory_iterator(path),
            std::filesystem::directory_iterator(),
            std::back_inserter(imgs_in_directory));
  std::sort(imgs_in_directory.begin(), imgs_in_directory.end());

  // 2. Set classes:
  // Brute Force Matcher
  auto bf = cv::BFMatcher(cv::NORM_HAMMING, true);

  // 3. Initialize variables
  auto prev_frame = proto_recon::Frame{};
  auto current_frame = proto_recon::Frame{};
  std::vector<cv::DMatch> matches;
  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  std::vector<cv::Mat> trajectory;
  cv::Mat trajectory_img =
      cv::Mat::zeros(600, 600, CV_8UC3);  // For trajectory visualization

  // 4. Set calibration matrix
  const cv::Mat K =
      (cv::Mat_<double>(3, 3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);

  // 5. Iterate over the frames
  uint64_t frame_id = 0;
  for (const auto& img_path : imgs_in_directory) {
    const auto img = cv::imread(img_path.string());
    current_frame = proto_recon::Frame(frame_id, frame_id, img);

    if (prev_frame.id() == -1) {
      prev_frame = current_frame;
      prev_frame.extractFeatures();
      continue;
    }

    // Feature detection in current frame
    current_frame.extractFeatures();

    // Match features using brute force
    bf.match(prev_frame.descriptors(), current_frame.descriptors(), matches);
    // Sort matches by distance
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& m1, const cv::DMatch& m2) {
                return m1.distance < m2.distance;
              });
    // Extract matched keypoints
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> current_pts;
    for (const auto& match : matches) {
      prev_pts.push_back(prev_frame.keypoints()[match.queryIdx].pt);
      current_pts.push_back(current_frame.keypoints()[match.trainIdx].pt);
    }

    // Estimate essential matrix
    cv::Mat E = cv::findEssentialMat(prev_pts, current_pts, K, cv::RANSAC,
                                     0.999, 1.0, cv::noArray());

    // Recover pose
    cv::Mat R;
    cv::Mat t;
    cv::recoverPose(E, prev_pts, current_pts, R, t);

    // Normalize translation vector to maintain consistent scale
    double curr_t_magnitude = cv::norm(t);
    if (curr_t_magnitude > 0) {
      double scale = 10.0 / curr_t_magnitude;
      t *= scale;
    }

    // Update the pose
    cv::Mat curr_pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(curr_pose(cv::Rect(0, 0, 3, 3)));
    t.copyTo(curr_pose(cv::Rect(3, 0, 1, 3)));
    pose *= curr_pose;

    // Draw trajectory
    double x = pose.at<double>(0, 3);
    double z = pose.at<double>(2, 3);
    cv::circle(trajectory_img,
               cv::Point(300 + static_cast<int>(x), 300 - static_cast<int>(z)),
               1, cv::Scalar(0, 255, 0), 1);

    // Display results
    cv::imshow("Trajectory", trajectory_img);
    cv::imshow("Frame", current_frame.img());
    trajectory.push_back(pose.clone());

    // Update previous frame
    prev_frame = current_frame;
    ++frame_id;

    if (cv::waitKey(1) == 'q') {
      break;
    }
  }

  cv::destroyAllWindows();
  writeTrajectory(imgs_in_directory, trajectory);
  proto_recon::drawTrajectory(trajectory);
  return 0;
}