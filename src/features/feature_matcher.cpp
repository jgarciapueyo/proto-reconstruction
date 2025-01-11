#include "proto_recon/features/feature_matcher.h"

namespace proto_recon {

FeatureMatcher::FeatureMatcher() : bf_(cv::NORM_HAMMING) {}

void FeatureMatcher::match(const std::shared_ptr<Frame>& prev_frame_,
                           const std::shared_ptr<Frame>& current_frame_,
                           std::vector<cv::DMatch>& good_matches) const {
  std::vector<std::vector<cv::DMatch>> matches;
  auto prev_frame_desc = prev_frame_->descriptors();
  auto curr_frame_desc = current_frame_->descriptors();

  bf_.knnMatch(prev_frame_->descriptors(), current_frame_->descriptors(),
               matches, 2);

  // Ratio test matches
  good_matches = {};
  for (const auto& best_two_matches : matches) {
    auto best_match = best_two_matches[0];
    auto second_best_match = best_two_matches[1];

    if ((best_match.distance < threshold_distance_) &&
        (best_match.distance < second_best_match.distance * 0.75)) {
      good_matches.push_back(best_match);
    }
  }
}

void FeatureMatcher::extract_matched_keypoints(
    const std::shared_ptr<Frame>& prev_frame_,
    const std::shared_ptr<Frame>& current_frame_,
    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& prev_pts,
    std::vector<cv::Point2f>& curr_pts) {
  prev_pts.clear();
  curr_pts.clear();

  for (const auto& match : matches) {
    prev_pts.push_back(prev_frame_->keypoints()[match.queryIdx].pt);
    curr_pts.push_back(current_frame_->keypoints()[match.trainIdx].pt);
  }
}

}  // namespace proto_recon