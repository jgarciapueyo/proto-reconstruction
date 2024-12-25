#include <filesystem>
#include <opencv2/opencv.hpp>

int main() {
  // 1. Read path of frames
  const std::string path{"../data/rgbd_dataset_freiburg1_xyz/rgb"};
  std::vector<std::filesystem::path> frames_in_directory;
  std::copy(std::filesystem::directory_iterator(path),
            std::filesystem::directory_iterator(),
            std::back_inserter(frames_in_directory));
  std::sort(frames_in_directory.begin(), frames_in_directory.end());

  // 2. Display the images
  for (const auto& frame_path : frames_in_directory) {
    auto current_frame = cv::imread(frame_path.string());
    cv::imshow("Display Frame", current_frame);
    cv::waitKey(0);
  }

  cv::destroyAllWindows();
  return 0;
}