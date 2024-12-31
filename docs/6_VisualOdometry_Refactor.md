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