#include "proto_recon/vo/frame.h"

#include <opencv2/features2d.hpp>
#include <utility>

namespace proto_recon {

Frame::Frame() : id_(-1), timestamp_(-1) {}

Frame::Frame(const uint64_t id, const double timestamp, cv::Mat img)
    : id_(id), timestamp_(timestamp), img_(std::move(img)) {}

void Frame::extractFeatures() {
  // TODO: improve this so that it does not instantiate every time
  const auto orb = cv::ORB::create(1000);
  orb->detectAndCompute(img_, cv::noArray(), kp_, desc_);
  // Reserve memory for each keypoint that can potentially become a map point
  map_points_.resize(kp_.size());
}

uint64_t Frame::id() const { return id_; }

double Frame::timestamp() const { return timestamp_; }

cv::Mat Frame::img() const { return img_; }

const std::vector<cv::KeyPoint>& Frame::keypoints() const { return kp_; }

const cv::Mat& Frame::descriptors() const { return desc_; }

const Sophus::SE3f& Frame::Tcw() const { return Tcw_; }

void Frame::setTcw(const Sophus::SE3f& Tcw) { Tcw_ = Tcw; }

void Frame::setMapPoint(const int idx,
                        const std::shared_ptr<MapPoint>& map_point) {
  map_points_[idx] = map_point;
}

}  // namespace proto_recon