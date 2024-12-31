#include <string>

#include "proto_recon/utils/imagestream.h"
#include "proto_recon/vo/vo.h"

int main() {
  std::string path{"../data/rgbd_dataset_freiburg1_xyz/rgb"};
  const proto_recon::ImageStream image_stream(path);
  proto_recon::VisualOdometry vo(image_stream);

  vo.configure();
  vo.run();

  return 0;
}