#include "proto_recon/vo/vo.h"

#include <Eigen/Core>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <utility>

namespace proto_recon {

VisualOdometry::VisualOdometry(ImageStream image_stream)
    : image_stream_(std::move(image_stream)), current_pose_(Sophus::SE3f()) {
  map_ = std::make_shared<Map>();
  map_visualizer_ = std::make_shared<MapVisualizer>(map_);
  cv::eigen2cv(K_, K);
}

bool VisualOdometry::configure() { return true; }

void VisualOdometry::run() {
  while (!image_stream_.finished()) {
    bool step_is_correct = step();
    if (!step_is_correct) {
      break;
    }
  }
  std::cout << "End" << std::endl;
}

bool VisualOdometry::step() {
  auto [idx, img] = image_stream_.nextImg();
  std::cout << "Frame: " << idx << std::endl;
  const auto frame = std::make_shared<Frame>(idx, idx, img);
  processFrame(frame);
  map_visualizer_->update();
  return true;
}

void VisualOdometry::processFrame(std::shared_ptr<Frame> frame) {
  current_frame_ = frame;

  if (!first_frame_processed_) {
    prev_frame_ = current_frame_;
    prev_frame_->extractFeatures();
    first_frame_processed_ = true;
    return;
  }

  current_frame_->extractFeatures();

  // Match features using brute force
  // TODO: do not create each time
  auto bf = cv::BFMatcher(cv::NORM_HAMMING, true);
  // TODO: do not create each time
  std::vector<cv::DMatch> matches;

  bf.match(prev_frame_->descriptors(), current_frame_->descriptors(), matches);
  // Sort matches by distance
  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch& m1, const cv::DMatch& m2) {
              return m1.distance < m2.distance;
            });
  // Extract matched keypoints
  std::vector<cv::Point2f> prev_pts;
  std::vector<cv::Point2f> current_pts;
  for (const auto& match : matches) {
    prev_pts.push_back(prev_frame_->keypoints()[match.queryIdx].pt);
    current_pts.push_back(current_frame_->keypoints()[match.trainIdx].pt);
  }

  // Estimate essential matrix
  cv::Mat E = cv::findEssentialMat(prev_pts, current_pts, K, cv::RANSAC, 0.999,
                                   1.0, cv::noArray());

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
  Sophus::SE3f T(R_sophus, t_sophus);
  current_pose_ *= T;
  current_frame_->setTcw(current_pose_);
  map_->insertKeyframe(current_frame_);

  // Compute map points
  Eigen::Matrix<float, 3, 4> prev_frame_proj_matrix =
      K_ * prev_frame_->Tcw().matrix3x4();
  cv::Mat prev_frame_proj_matrix_cv;
  cv::eigen2cv(prev_frame_proj_matrix, prev_frame_proj_matrix_cv);

  Eigen::Matrix<float, 3, 4> current_frame_proj_matrix =
      K_ * current_frame_->Tcw().matrix3x4();
  cv::Mat current_frame_proj_matrix_cv;
  cv::eigen2cv(current_frame_proj_matrix, current_frame_proj_matrix_cv);

  cv::Mat points_in_3d;
  cv::triangulatePoints(prev_frame_proj_matrix_cv, current_frame_proj_matrix_cv,
                        prev_pts, current_pts, points_in_3d);

  for (int points_in_3d_idx = 0; points_in_3d_idx < points_in_3d.cols;
       ++points_in_3d_idx) {
    cv::Mat x = points_in_3d.col(points_in_3d_idx);
    // x /= x.at<double>(3, 0);
    double scale = x.at<double>(3, 0);
    const auto point = Eigen::Vector3f(x.at<float>(0, 0), x.at<float>(1, 0),
                                 x.at<float>(2, 0));
    auto mappoint = std::make_shared<MapPoint>(point);
    map_->insertMapPoint(mappoint);
  }

  prev_frame_ = current_frame_;
}

}  // namespace proto_recon