#include "proto_recon/visualization/visualizer.h"

#include <pangolin/pangolin.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <vector>

void drawTrajectory(const std::vector<cv::Mat>& trajectory) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

  // Create Interactive View in window
  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0F / 768.0F)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0F, 1.0F, 1.0F, 1.0F);
    glLineWidth(2);
    for (size_t i = 0; i < trajectory.size(); i++) {
      // draw three axes of each pose
      // Vector3d Ow = poses[i].translation();
      // Vector3d Xw = poses[i] ∗ (0.1 ∗ Vector3d(1, 0, 0));
      // Vector3d Yw = poses[i] ∗ (0.1 ∗ Vector3d(0, 1, 0));
      // Vector3d Zw = poses[i] ∗ (0.1 ∗ Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(trajectory[i].at<double>(0, 3), trajectory[i].at<double>(1, 3),
                 trajectory[i].at<double>(2, 3));
      glVertex3d(trajectory[i].at<double>(0, 3) + 0.1,
                 trajectory[i].at<double>(1, 3),
                 trajectory[i].at<double>(2, 3));
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(trajectory[i].at<double>(0, 3), trajectory[i].at<double>(1, 3),
                 trajectory[i].at<double>(2, 3));
      glVertex3d(trajectory[i].at<double>(0, 3),
                 trajectory[i].at<double>(1, 3) + 0.1,
                 trajectory[i].at<double>(2, 3));
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(trajectory[i].at<double>(0, 3), trajectory[i].at<double>(1, 3),
                 trajectory[i].at<double>(2, 3));
      glVertex3d(trajectory[i].at<double>(0, 3), trajectory[i].at<double>(1, 3),
                 trajectory[i].at<double>(2, 3) + 0.1);
      glEnd();
    }

    // draw a connection
    for (size_t i = 0; i < trajectory.size() - 1; i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = trajectory[i], p2 = trajectory[i + 1];
      glVertex3d(p1.at<double>(0, 3), p1.at<double>(1, 3), p1.at<double>(2, 3));
      glVertex3d(p2.at<double>(0, 3), p2.at<double>(1, 3), p2.at<double>(2, 3));
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);
    // sleep 5 ms
  }
}