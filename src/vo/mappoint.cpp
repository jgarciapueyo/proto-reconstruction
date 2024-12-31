#include "proto_recon/vo/mappoint.h"

#include <utility>

namespace proto_recon {

uint64_t MapPoint::next_id_ = 0;

MapPoint::MapPoint(Eigen::Vector3f position3D)
    : position3D_(std::move(position3D)) {
  id_ = next_id_;
  next_id_++;
}

uint64_t MapPoint::id() const { return id_; }

Eigen::Vector3f MapPoint::position() const { return position3D_; }

}  // namespace proto_recon