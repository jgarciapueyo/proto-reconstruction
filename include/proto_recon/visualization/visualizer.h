#pragma once

#include <sophus/se3.hpp>
#include <vector>

#include "proto_recon/vo/mappoint.h"

namespace proto_recon {

void drawTrajectory(const std::vector<Sophus::SE3f>& trajectory,
                    const std::vector<MapPoint>& mappoints);

}