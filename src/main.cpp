#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "proto_recon/visualization/visualizer.h"
#include "proto_recon/vo/frame.h"

static void writeTrajectory(
    const std::vector<std::filesystem::path>& frames_in_directory,
    const std::vector<Sophus::SE3f>& trajectory) {
  std::ofstream trajectory_file("estimated_trajectory_sophus.txt");
  trajectory_file << "# estimated trajectory" << std::endl;
  trajectory_file << "# file: 'rgbd_dataset_freiburg1_xyz.bag'" << std::endl;
  trajectory_file << "# timestamp tx ty tz qx qy qz qw" << std::endl;

  for (int idx = 0; idx < frames_in_directory.size() - 1; ++idx) {
    const auto& frame_path = frames_in_directory[idx];
    auto timestamp_str = frame_path.stem().string();
    auto pose = trajectory[idx];
    auto quat = pose.so3().unit_quaternion();
    trajectory_file << timestamp_str << " " << pose.translation().x() << " "
                    << pose.translation().y() << " " << pose.translation().z()
                    << " " << quat.x() << " " << quat.y() << " " << quat.z()
                    << " " << quat.w() << std::endl;
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
  auto pose = Sophus::SE3f();
  std::vector<Sophus::SE3f> trajectory;
  cv::Mat trajectory_img =
      cv::Mat::zeros(600, 600, CV_8UC3);  // For trajectory visualization
  std::vector<proto_recon::MapPoint> mappoints;

  // 4. Set calibration matrix
  const cv::Mat K =
      (cv::Mat_<double>(3, 3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);
  const Eigen::Matrix3f K_{{520.9, 0, 325.1}, {0, 521.0, 249.7}, {0, 0, 1}};

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
      double scale = 1.0 / curr_t_magnitude;
      t *= scale;
    }

    // Update the pose
    Sophus::Matrix3f R_sophus;
    cv::cv2eigen(R, R_sophus);
    Sophus::Vector3f t_sophus;
    cv::cv2eigen(t, t_sophus);
    Sophus::SE3f curr_pose(R_sophus, t_sophus);
    pose *= curr_pose;
    current_frame.setTcw(pose);

    // Compute map points
    Eigen::Matrix<float, 3, 4> prev_frame_proj_matrix =
        K_ * prev_frame.Tcw().matrix3x4();
    cv::Mat prev_frame_proj_matrix_cv;
    cv::eigen2cv(prev_frame_proj_matrix, prev_frame_proj_matrix_cv);

    Eigen::Matrix<float, 3, 4> current_frame_proj_matrix =
        K_ * current_frame.Tcw().matrix3x4();
    cv::Mat current_frame_proj_matrix_cv;
    cv::eigen2cv(current_frame_proj_matrix, current_frame_proj_matrix_cv);

    cv::Mat points_in_3d;
    cv::triangulatePoints(prev_frame_proj_matrix_cv,
                          current_frame_proj_matrix_cv, prev_pts,
                          current_pts, points_in_3d);

    for (int points_in_3d_idx = 0; points_in_3d_idx < points_in_3d.cols;
         ++points_in_3d_idx) {
      cv::Mat x = points_in_3d.col(points_in_3d_idx);
      // x /= x.at<double>(3, 0);
      double scale = x.at<double>(3, 0);
      mappoints.emplace_back(Eigen::Vector3f(
          x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0)));
    }

    // Draw trajectory
    double x = pose.translation().x();
    double z = pose.translation().z();
    cv::circle(trajectory_img,
               cv::Point(300 + static_cast<int>(x), 300 - static_cast<int>(z)),
               1, cv::Scalar(0, 255, 0), 1);

    // Display results
    cv::imshow("Trajectory", trajectory_img);
    cv::imshow("Frame", current_frame.img());
    trajectory.push_back(pose);

    // Update previous frame
    prev_frame = current_frame;
    ++frame_id;

    if (cv::waitKey(1) == 'q') {
      break;
    }
  }

  cv::destroyAllWindows();
  writeTrajectory(imgs_in_directory, trajectory);
  proto_recon::drawTrajectory(trajectory, mappoints);
  std::cout << mappoints.size() << std::endl;
  return 0;
}