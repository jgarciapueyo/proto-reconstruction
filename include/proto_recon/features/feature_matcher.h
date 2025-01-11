#pragma once

#include <memory>
#include <opencv2/features2d.hpp>

#include "proto_recon/vo/frame.h"

namespace proto_recon {

class FeatureMatcher {
 public:
  FeatureMatcher();
  void match(const std::shared_ptr<Frame>& prev_frame_,
             const std::shared_ptr<Frame>& current_frame_,
             std::vector<cv::DMatch>& good_matches) const;

  static void extract_matched_keypoints(
      const std::shared_ptr<Frame>& prev_frame_,
      const std::shared_ptr<Frame>& current_frame_,
      const std::vector<cv::DMatch>& matches,
      std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& curr_pts);

 private:
  cv::BFMatcher bf_;
  int threshold_distance_ = 20;
  float min_ratio_distance_ = 0.8;
};

}  // namespace proto_recon