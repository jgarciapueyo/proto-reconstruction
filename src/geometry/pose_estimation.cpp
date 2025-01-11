#include "proto_recon/geometry/pose_estimation.h"

#include <opencv2/core/eigen.hpp>

namespace proto_recon {

void PoseEstimation::computePoseFromMatches(
    const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K, cv::Mat& E, Sophus::SE3f& Tcw) {
  // Estimate essential matrix
  E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1,
                           cv::noArray());
  // Estimate pose
  cv::Mat R;
  cv::Mat t;
  cv::recoverPose(E, pts1, pts2, R, t);

  // Compose the pose
  Sophus::Matrix3f R_sophus;
  cv::cv2eigen(R, R_sophus);
  Sophus::Vector3f t_sophus;
  cv::cv2eigen(t, t_sophus);
  t_sophus.normalize();
  Tcw = Sophus::SE3f(R_sophus, -t_sophus);
}

}  // namespace proto_recon