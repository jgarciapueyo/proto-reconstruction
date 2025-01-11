# Part 6: Refactor the Visual Odometry
We are going to continue with the refactor of the modules so that each component has its own responsibility and it is easier to modify and test.

## Step 1: Feature Matcher class
We are going to create a module for feature matching each image of a frame. We create the class `include/proto_recon/features/feature_matcher.h`:
```c++
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
```
and its implementation in `src/features/feature_matcher.cpp`:
```c++
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
```

## Step 2: Pose Estimation class
We are going to create a module for estimating the position given the keypoints associated to features between a pair of images. We create the class `include/proto_recon/geometry/pose_estimation.h`:
```c++
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
```
and its implementation
```c++
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
```

## Step 3: Modify Visual Odometry class
In the class `VisualOdometry` we are going to modify the method `configure()` to pass the calibration matrix:
```c++
// vo.h
  bool configure(const Eigen::Matrix3f& K);

// vo.cpp
bool VisualOdometry::configure(const Eigen::Matrix3f& K) {
  K_ = K;
  cv::eigen2cv(K_, K_mat_);
  return true;
}
```
and the `processFrame()` method to use the Feature Matcher and Position Estimation modules:
```c++
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
```

## Step 4: Add option to draw Map Points
In the MapVisualizer class, we are going to add a method to draw the map points:
```c++
// include/proto_recon/visualization/mapvisualizer.h
  void drawMapPoints() const;

// src/visualization/mapvisualizer.h
void MapVisualizer::drawMapPoints() const {
  glPointSize(2);
  glBegin(GL_POINTS);
  glColor3f(0.0, 0.0, 0.0);
  for (const auto& [mp_id, mappoint] : map_->map_points()) {
    glVertex3f(mappoint->position().x(), mappoint->position().y(),
               mappoint->position().z());
  }
  glEnd();
}
```

## Step 5: Create a library
We are going to modify our executable, so our implementation is a library that we can link programs against. For that, we create a `src/CMakeLists.txt`:
```cmake
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(proto_reconstruction SHARED
        features/feature_matcher.cpp
        geometry/pose_estimation.cpp
        utils/imagestream.cpp
        visualization/mapvisualizer.cpp
        vo/frame.cpp
        vo/map.cpp
        vo/mappoint.cpp
        vo/vo.cpp
)
target_link_libraries(proto_reconstruction ${THIRD_PARTY_LIBS})
```
We also have to modify the initial `CMakeLists.txt` in the root path:
```cmake
...
set(THIRD_PARTY_LIBS
Eigen3::Eigen
${OpenCV_LIBS}
${Sophus_LIBRARIES}
${Pangolin_LIBRARIES}
)

add_subdirectory(src)
add_subdirectory(apps)
```
Finally, we create a folder `apps/` in the root path where we will create our executables. There, we add `apps/CMakeLists.txt`:
```cmake
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(main_vo main_vo.cpp)
target_link_libraries(main_vo proto_reconstruction ${THIRD_PARTY_LIBS})
```
that allows to create main program, like the one in `apps/main_vo.cpp`:
```c++
#include <string>

#include <proto_recon/utils/imagestream.h>
#include <proto_recon/vo/vo.h>

int main() {
  std::string path{"../../data/rgbd_dataset_freiburg1_xyz/rgb"};
  const proto_recon::ImageStream image_stream(path);
  proto_recon::VisualOdometry vo(image_stream);

  const Eigen::Matrix3f K{{517.3, 0, 318.6}, {0, 516.5, 255.3}, {0, 0, 1}}; // rgbd

  vo.configure(K);
  vo.run();

  return 0;
}
```