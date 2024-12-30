#pragma once

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <vector>

#include "proto_recon/vo/mappoint.h"

namespace proto_recon {

class Frame {
 public:
  Frame();
  Frame(uint64_t id, double timestamp, cv::Mat img);
  uint64_t id() const;
  double timestamp() const;
  cv::Mat img() const;
  const std::vector<cv::KeyPoint>& keypoints() const;
  const cv::Mat& descriptors() const;
  const Sophus::SE3f& Tcw() const;
  void setTcw(const Sophus::SE3f& Tcw);
  void setMapPoint(int idx, const std::shared_ptr<MapPoint>& map_point);
  void extractFeatures();

 private:
  uint64_t id_;
  double timestamp_;
  cv::Mat img_;
  std::vector<cv::KeyPoint> kp_;
  cv::Mat desc_;
  std::vector<std::shared_ptr<MapPoint>> map_points_;
  Sophus::SE3f Tcw_;  // Pose of the frame
};

}  // namespace proto_recon