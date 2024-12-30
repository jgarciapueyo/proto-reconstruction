#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace proto_recon {

class MapPoint {
 public:
  explicit MapPoint(Eigen::Vector3f position3D);
  uint64_t id() const;

 private:
  uint64_t id_;
  Eigen::Vector3f position3D_;
  cv::Mat desc_;  // Most distinct descriptor of the MapPoint

  static uint64_t next_id_;
};

}  // namespace proto_recon