#pragma once
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace proto_recon {

class PoseEstimation {
 public:
  PoseEstimation() = default;
  void computePoseFromMatches(const std::vector<cv::Point2f>& pts1,
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& K, cv::Mat& E, Sophus::SE3f& Tcw);
};

}  // namespace proto_recon