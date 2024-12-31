# Part 6: Refactor the Visual Odometry
We have the components for visual odometry, including estimation of the poses and map points, and the visualization. However, there is some bug in the triangulation of the mappoints, the estimation of the scale is done arbitrarily and we perform no bundle adjustment to refine the poses. Before trying to fix the bug or improve the code, we are going to refactor so that it is easier to modify.

## Step 1: Image Stream class
First, we need to abstract the logic about reading images from the dataset in `include/proto_recon/utils/imagestream.h`:
```c++
#pragma once

#include <filesystem>
#include <opencv2/core.hpp>

namespace proto_recon {

class ImageStream {
 public:
  explicit ImageStream(std::string directory_path);
  std::tuple<size_t, cv::Mat> nextImg();
  bool finished() const;

 private:
  std::string directory_path_;
  std::vector<std::filesystem::path> files_in_directory_;
  size_t img_idx_;
};

}  // namespace proto_recon
```
and its implementation in `src/utils/imagestream.cpp`:
```c++
#include "proto_recon/utils/imagestream.h"

#include <opencv2/imgcodecs.hpp>
#include <utility>

namespace proto_recon {

ImageStream::ImageStream(std::string directory_path)
    : directory_path_(std::move(directory_path)), img_idx_(0) {
  std::copy(std::filesystem::directory_iterator(directory_path_),
            std::filesystem::directory_iterator(),
            std::back_inserter(files_in_directory_));
  std::sort(files_in_directory_.begin(), files_in_directory_.end());
}

std::tuple<size_t, cv::Mat> ImageStream::nextImg() {
  cv::Mat img = cv::imread(files_in_directory_[img_idx_]);
  ++img_idx_;
  return {img_idx_ - 1, img};
}

bool ImageStream::finished() const {
  return img_idx_ == files_in_directory_.size();
}

}  // namespace proto_recon
```

# Step 2: Map class
We need a map class that will store all the information about all the frames (which already contain its pose) and the map points. This will be in `include/proto_recon/vo/map.h`:
```c++
#pragma once

#include <memory>

#include "proto_recon/vo/frame.h"
#include "proto_recon/vo/mappoint.h"

namespace proto_recon {

using ID = uint64_t;

class Map {
 public:
  Map();
  void insertKeyframe(const std::shared_ptr<Frame>& keyframe);
  void insertMapPoint(const std::shared_ptr<MapPoint>& map_point);
  const std::unordered_map<ID, std::shared_ptr<Frame>>& keyframes() const;
  const std::unordered_map<ID, std::shared_ptr<MapPoint>>& map_points() const;

 private:
  std::unordered_map<ID, std::shared_ptr<Frame>> keyframes_;
  std::unordered_map<ID, std::shared_ptr<MapPoint>> map_points_;
};

}  // namespace proto_recon
```
and in `src/vo/map.cpp`:
````c++
#include "proto_recon/vo/map.h"

namespace proto_recon {

Map::Map() = default;

void Map::insertKeyframe(const std::shared_ptr<Frame>& keyframe) {
  keyframes_[keyframe->id()] = keyframe;
}

void Map::insertMapPoint(const std::shared_ptr<MapPoint>& map_point) {
  map_points_[map_point->id()] = map_point;
}

const std::unordered_map<ID, std::shared_ptr<Frame>>& Map::keyframes() const {
  return keyframes_;
}

const std::unordered_map<ID, std::shared_ptr<MapPoint>>& Map::map_points()
    const {
  return map_points_;
}

}  // namespace proto_recon
````

## Part 3: Create Visual Odometry class
We are going to encapsulate all the logic inside the `VisualOdometry` class so that the main function is small. In file `include/vo/vo.h`:
```c++
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

  ImageStream image_stream_;

  std::shared_ptr<Map> map_;
  std::shared_ptr<MapVisualizer> map_visualizer_;
};

}  // namespace proto_recon
```
We see how it contains the image stream from which it will read the images, a pointer to the map containing the keyframes and map points and a pointer to the map visualizer, to separate the map container from the logic to visualize it. The implementation is in `src/vo/vo.cpp`:
```c++
#include "proto_recon/vo/vo.h"

#include <iostream>
#include <utility>

namespace proto_recon {

VisualOdometry::VisualOdometry(ImageStream image_stream)
    : image_stream_(std::move(image_stream)) {
  map_ = std::make_shared<Map>();
  map_visualizer_ = std::make_shared<MapVisualizer>(map_);
}

bool VisualOdometry::configure() { return true; }

void VisualOdometry::run() {
  while (!image_stream_.finished()) {
    bool step_is_correct = step();
    if (!step_is_correct) {
      break;
    }
  }
  std::cout << "End" << std::endl;
}

bool VisualOdometry::step() {
  auto [idx, img] = image_stream_.nextImg();
  const auto frame = std::make_shared<Frame>(idx, idx, img);
  std::cout << "Frame: " << frame->id() << std::endl;
  // TODO: logic for estimating movement of camera
  map_visualizer_->update();
  return true;
}

}  // namespace proto_recon
```
Which contains logic about initialization in the constructor, a `run()` function to start performing Structure-from-motion estimation while there are images and the `step()` function which reads the next image, creates the frame and performs the logic (still missing). Finally, we call update to update the visualization of the map.

# Part 4: Update the map visualizer
We are going to rename the file `include/proto_recon/visualization/visualizer.h` to `include/proto_recon/visualization/mapvisualizer.h` to create a class that allows us to visualize the map easily:
```c++
class MapVisualizer {
 public:
  explicit MapVisualizer(const std::shared_ptr<proto_recon::Map>& map);
  void update() const;

 private:
  void drawKeyFrames() const;

  std::shared_ptr<Map> map_;

  // Need to store state of the Pangolin visualizer
  pangolin::OpenGlRenderState s_cam_;
  pangolin::View* d_cam_;
};
```
and its implementation
```c++

namespace proto_recon {

MapVisualizer::MapVisualizer(const std::shared_ptr<proto_recon::Map>& map)
    : map_(map) {
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  s_cam_ = pangolin::OpenGlRenderState(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.001, 10000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

  // Create Interactive View in window
  d_cam_ = &pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0F / 768.0F)
                .SetHandler(new pangolin::Handler3D(s_cam_));
}

void MapVisualizer::update() const {
  std::cout << "MapVisualizer::update " << std::endl;
  std::cout << map_->keyframes().size() << std::endl;
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  d_cam_->Activate(s_cam_);
  glClearColor(1.0F, 1.0F, 1.0F, 1.0F);
  drawKeyFrames();
  pangolin::FinishFrame();
}

void MapVisualizer::drawKeyFrames() const {
  glLineWidth(2);

  for (auto& [keyframe_id, keyframe] : map_->keyframes()) {
    // draw three axes of each pose
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glVertex3d(keyframe->Tcw().translation().x() + 0.1,
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y() + 0.1,
               keyframe->Tcw().translation().z());
    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z() + 0.1);
    glEnd();
  }

  // draw a connection
  bool first_frame = true;
  std::shared_ptr<Frame> prev_frame;
  for (auto it = map_->keyframes().begin(); it != map_->keyframes().end();
       ++it) {
    if (first_frame) {
      prev_frame = it->second;
      first_frame = false;
    }

    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3d(prev_frame->Tcw().translation().x(),
               prev_frame->Tcw().translation().y(),
               prev_frame->Tcw().translation().z());
    glVertex3d(it->second->Tcw().translation().x(),
               it->second->Tcw().translation().y(),
               it->second->Tcw().translation().z());
    glEnd();

    prev_frame = it->second;
  }

  usleep(5000);
  // sleep 5 ms
}

}
```

## Part 5: Move logic inside Visual Odometry class
Now, we are going to move all the logic inside the `main()` function to `VisualOdometry::processFrame(std::shared_ptr<Frame> frame)`:
```c++
 void VisualOdometry::processFrame(std::shared_ptr<Frame> frame) {
  current_frame_ = frame;

  if (!first_frame_processed_) {
    prev_frame_ = current_frame_;
    prev_frame_->extractFeatures();
    first_frame_processed_ = true;
    return;
  }

  current_frame_->extractFeatures();

  // Match features using brute force
  // TODO: do not create each time
  auto bf = cv::BFMatcher(cv::NORM_HAMMING, true);
  // TODO: do not create each time
  std::vector<cv::DMatch> matches;

  bf.match(prev_frame_->descriptors(), current_frame_->descriptors(), matches);
  // Sort matches by distance
  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch& m1, const cv::DMatch& m2) {
              return m1.distance < m2.distance;
            });
  // Extract matched keypoints
  std::vector<cv::Point2f> prev_pts;
  std::vector<cv::Point2f> current_pts;
  for (const auto& match : matches) {
    prev_pts.push_back(prev_frame_->keypoints()[match.queryIdx].pt);
    current_pts.push_back(current_frame_->keypoints()[match.trainIdx].pt);
  }

  // Estimate essential matrix
  cv::Mat E = cv::findEssentialMat(prev_pts, current_pts, K, cv::RANSAC, 0.999,
                                   1.0, cv::noArray());

  // Recover pose
  cv::Mat R;
  cv::Mat t;
  cv::recoverPose(E, prev_pts, current_pts, R, t);

  // Normalize translation vector to maintain consistent scale
  double curr_t_magnitude = cv::norm(t);
  if (curr_t_magnitude > 0) {
    double scale = 1.0 / curr_t_magnitude;
    t *= scale;
  }

  // Update the pose
  Sophus::Matrix3f R_sophus;
  cv::cv2eigen(R, R_sophus);
  Sophus::Vector3f t_sophus;
  cv::cv2eigen(t, t_sophus);
  Sophus::SE3f T(R_sophus, t_sophus);
  current_pose_ *= T;
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
    auto mappoint = std::make_shared<MapPoint>(point);
    map_->insertMapPoint(mappoint);
  }

  prev_frame_ = current_frame_;
}
```
