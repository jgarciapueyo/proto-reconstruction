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