#pragma once

#include "proto_recon/features/feature_matcher.h"
#include "proto_recon/geometry/pose_estimation.h"
#include "proto_recon/utils/imagestream.h"
#include "proto_recon/visualization/mapvisualizer.h"
#include "proto_recon/vo/map.h"

namespace proto_recon {

class VisualOdometry {
 public:
  explicit VisualOdometry(ImageStream image_stream);
  bool configure(const Eigen::Matrix3f& K);
  void run();

 private:
  bool step();
  void processFrame(std::shared_ptr<Frame> frame);

  ImageStream image_stream_;
  FeatureMatcher feature_matcher_;
  PoseEstimation pose_estimation_;

  std::shared_ptr<Map> map_;
  std::shared_ptr<MapVisualizer> map_visualizer_;

  // Needed for tracking
  bool first_frame_processed_ = false;
  std::shared_ptr<Frame> prev_frame_, current_frame_;
  Sophus::SE3f current_pose_;
  Eigen::Matrix3f K_;
  cv::Mat K_mat_;
};

}  // namespace proto_recon