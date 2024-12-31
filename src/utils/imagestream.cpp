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

std::vector<std::filesystem::path> ImageStream::filenames() const {
  return files_in_directory_;
}

}  // namespace proto_recon