#pragma once

#include <filesystem>
#include <opencv2/core.hpp>

namespace proto_recon {

class ImageStream {
 public:
  explicit ImageStream(std::string directory_path);
  std::tuple<size_t, cv::Mat> nextImg();
  bool finished() const;
  std::vector<std::filesystem::path> filenames() const;

 private:
  std::string directory_path_;
  std::vector<std::filesystem::path> files_in_directory_;
  size_t img_idx_;
};

}  // namespace proto_recon