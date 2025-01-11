#include <string>

#include <proto_recon/utils/imagestream.h>
#include <proto_recon/vo/vo.h>

int main() {
  std::string path{"../../data/kitti_dataset/01/image_0"};
  // std::string path{"../../data/rgbd_dataset_freiburg1_xyz/rgb"};
  const proto_recon::ImageStream image_stream(path);
  proto_recon::VisualOdometry vo(image_stream);

  // const Eigen::Matrix3f K{{517.3, 0, 318.6}, {0, 516.5, 255.3}, {0, 0, 1}}; // rgbd
  // Eigen::Matrix3f K{{458.653999999, 0, 367.214999999999975}, {0, 457.2959999999999923, 248.37500000}, {0, 0, 1}}; // test
  const Eigen::Matrix3f K{{718.856, 0, 607.1928}, {0, 718.856, 185.2157}, {0, 0, 1}}; // kitti
  vo.configure(K);
  vo.run();

  return 0;
}