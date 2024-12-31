#pragma once

#include <proto_recon/visualization/mapvisualizer.h>

#include "proto_recon/utils/imagestream.h"
#include "proto_recon/vo/map.h"

namespace proto_recon {

class VisualOdometry {
 public:
  explicit VisualOdometry(ImageStream image_stream);
  bool configure();
  void run();

 private:
  bool step();
  void processFrame(std::shared_ptr<Frame> frame);

  ImageStream image_stream_;

  std::shared_ptr<Map> map_;
  std::shared_ptr<MapVisualizer> map_visualizer_;

  // Needed for tracking
  bool first_frame_processed_ = false;
  std::shared_ptr<Frame> prev_frame_, current_frame_;
  Sophus::SE3f current_pose_;
  // const Eigen::Matrix3f K_{{520.9, 0, 325.1}, {0, 521.0, 249.7}, {0, 0, 1}};
  const Eigen::Matrix3f K_{{718.856, 0, 607.1928}, {0, 718.856, 185.2157}, {0, 0, 1}};
  cv::Mat K;
};

}  // namespace proto_recon