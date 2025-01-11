#pragma once

#include <pangolin/pangolin.h>

#include <sophus/se3.hpp>
#include <vector>

#include "proto_recon/vo/map.h"
#include "proto_recon/vo/mappoint.h"

namespace proto_recon {

class MapVisualizer {
 public:
  explicit MapVisualizer(const std::shared_ptr<proto_recon::Map>& map);
  void update() const;

 private:
  void drawKeyFrames() const;
  void drawMapPoints() const;

  std::shared_ptr<Map> map_;

  // Need to store state of the Pangolin visualizer
  pangolin::OpenGlRenderState s_cam_;
  pangolin::View* d_cam_;
};

void drawTrajectory(const std::vector<Sophus::SE3f>& trajectory,
                    const std::vector<MapPoint>& mappoints);

}  // namespace proto_recon