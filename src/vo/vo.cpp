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
}

bool VisualOdometry::configure(const Eigen::Matrix3f& K) {
  K_ = K;
  cv::eigen2cv(K_, K_mat_);
  return true;
}

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
  current_frame_->extractFeatures();

  if (!first_frame_processed_) {
    prev_frame_ = current_frame_;
    first_frame_processed_ = true;
    return;
  }

  // Match frames
  std::vector<cv::DMatch> matches;
  feature_matcher_.match(prev_frame_, current_frame_, matches);
  // Extract matched keypoints // TODO: make part of the class
  std::vector<cv::Point2f> prev_pts;
  std::vector<cv::Point2f> current_pts;
  proto_recon::FeatureMatcher::extract_matched_keypoints(
      prev_frame_, current_frame_, matches, prev_pts, current_pts);

  cv::Mat essential_matrix;
  Sophus::SE3f T_cw;
  pose_estimation_.computePoseFromMatches(prev_pts, current_pts, K_mat_,
                                          essential_matrix, T_cw);

  current_pose_ = T_cw * current_pose_;
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
    const auto point_in_world = prev_frame_->Tcw() * point;
    auto mappoint = std::make_shared<MapPoint>(point_in_world);
    map_->insertMapPoint(mappoint);
  }

  prev_frame_ = current_frame_;
}

}  // namespace proto_recon