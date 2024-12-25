#include <filesystem>
#include <opencv2/opencv.hpp>

int main() {
  int a = 5;
  // 1. Read path of frames
  const std::string path{"../data/rgbd_dataset_freiburg1_xyz/rgb"};
  std::vector<std::filesystem::path> frames_in_directory;
  std::copy(std::filesystem::directory_iterator(path),
            std::filesystem::directory_iterator(),
            std::back_inserter(frames_in_directory));
  std::sort(frames_in_directory.begin(), frames_in_directory.end());

  // 2. Set classes:
  // ORB feture detector and descriptor
  auto orb = cv::ORB::create(1000);
  // Brute Force Matcher
  auto bf = cv::BFMatcher(cv::NORM_HAMMING, true);

  // 3. Initialize variables
  auto prev_frame = cv::Mat{};
  std::vector<cv::KeyPoint> prev_keypoints;
  auto prev_descriptors = cv::Mat{};
  auto current_frame = cv::Mat{};
  std::vector<cv::KeyPoint> current_keypoints;
  auto current_descriptors = cv::Mat{};
  std::vector<cv::DMatch> matches;
  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat trajectory =
      cv::Mat::zeros(600, 600, CV_8UC3);  // For trajectory visualization

  // 4. Set calibration matrix
  const cv::Mat K =
      (cv::Mat_<double>(3, 3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);

  // 5. Iterate over the frames
  for (const auto& frame_path : frames_in_directory) {
    current_frame = cv::imread(frame_path.string());

    if (prev_frame.empty()) {
      prev_frame = current_frame;
      orb->detectAndCompute(prev_frame, cv::noArray(), prev_keypoints,
                            prev_descriptors);
      continue;
    }

    // Feature detection in current frame
    orb->detectAndCompute(current_frame, cv::noArray(), current_keypoints,
                          current_descriptors);

    // Match features using brute force
    bf.match(prev_descriptors, current_descriptors, matches);
    // Sort matches by distance
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& m1, const cv::DMatch& m2) {
                return m1.distance < m2.distance;
              });
    // Extract matched keypoints
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> current_pts;
    for (const auto& match : matches) {
      prev_pts.push_back(prev_keypoints[match.queryIdx].pt);
      current_pts.push_back(current_keypoints[match.trainIdx].pt);
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
    cv::circle(trajectory,
               cv::Point(300 + static_cast<int>(x), 300 - static_cast<int>(z)),
               1, cv::Scalar(0, 255, 0), 1);

    // Display results
    cv::imshow("Trajectory", trajectory);
    cv::imshow("Frame", current_frame);

    // Update previous frame
    prev_frame = current_frame;
    prev_keypoints = current_keypoints;
    prev_descriptors = current_descriptors;

    if (cv::waitKey(1) == 'q') {
      break;
    }
  }

  cv::destroyAllWindows();
  return 0;
}