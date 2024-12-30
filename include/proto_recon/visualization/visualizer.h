#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace proto_recon {

void drawTrajectory(const std::vector<cv::Mat>& trajectory);

}