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