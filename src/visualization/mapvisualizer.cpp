#include "proto_recon/visualization/mapvisualizer.h"

#include <pangolin/pangolin.h>
#include <unistd.h>

#include <vector>

namespace proto_recon {

MapVisualizer::MapVisualizer(const std::shared_ptr<proto_recon::Map>& map)
    : map_(map) {
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  s_cam_ = pangolin::OpenGlRenderState(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.001, 10000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

  // Create Interactive View in window
  d_cam_ = &pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0F / 768.0F)
                .SetHandler(new pangolin::Handler3D(s_cam_));
}

void MapVisualizer::update() const {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  d_cam_->Activate(s_cam_);
  glClearColor(1.0F, 1.0F, 1.0F, 1.0F);
  drawKeyFrames();
  pangolin::FinishFrame();
}

void MapVisualizer::drawKeyFrames() const {
  glLineWidth(2);

  for (auto& [keyframe_id, keyframe] : map_->keyframes()) {
    // draw three axes of each pose
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glVertex3d(keyframe->Tcw().translation().x() + 0.1,
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y() + 0.1,
               keyframe->Tcw().translation().z());
    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z());
    glVertex3d(keyframe->Tcw().translation().x(),
               keyframe->Tcw().translation().y(),
               keyframe->Tcw().translation().z() + 0.1);
    glEnd();
  }

  // draw a connection
  bool first_frame = true;
  std::shared_ptr<Frame> prev_frame;
  for (auto it = map_->keyframes().begin(); it != map_->keyframes().end();
       ++it) {
    if (first_frame) {
      prev_frame = it->second;
      first_frame = false;
    }

    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3d(prev_frame->Tcw().translation().x(),
               prev_frame->Tcw().translation().y(),
               prev_frame->Tcw().translation().z());
    glVertex3d(it->second->Tcw().translation().x(),
               it->second->Tcw().translation().y(),
               it->second->Tcw().translation().z());
    glEnd();

    prev_frame = it->second;
  }

  usleep(5000);
  // sleep 5 ms
}

void drawTrajectory(const std::vector<Sophus::SE3f>& trajectory,
                    const std::vector<MapPoint>& mappoints) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.001, 10000),
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
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(trajectory[i].translation().x(),
                 trajectory[i].translation().y(),
                 trajectory[i].translation().z());
      glVertex3d(trajectory[i].translation().x() + 0.1,
                 trajectory[i].translation().y(),
                 trajectory[i].translation().z());
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(trajectory[i].translation().x(),
                 trajectory[i].translation().y(),
                 trajectory[i].translation().z());
      glVertex3d(trajectory[i].translation().x(),
                 trajectory[i].translation().y() + 0.1,
                 trajectory[i].translation().z());
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(trajectory[i].translation().x(),
                 trajectory[i].translation().y(),
                 trajectory[i].translation().z());
      glVertex3d(trajectory[i].translation().x(),
                 trajectory[i].translation().y(),
                 trajectory[i].translation().z() + 0.1);
      glEnd();
    }

    // draw a connection
    for (size_t i = 0; i < trajectory.size() - 1; i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = trajectory[i], p2 = trajectory[i + 1];
      glVertex3d(p1.translation().x(), p1.translation().y(),
                 p1.translation().z());
      glVertex3d(p2.translation().x(), p2.translation().y(),
                 p2.translation().z());
      glEnd();
    }

    // draw mappoints
    glPointSize(3.0);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);
    for (const auto& p : mappoints) {
      // std::cout << "[" << p.position().x() << " " << p.position().y() << " "
      // << p.position().z() << "]" << std::endl;
      glVertex3f(p.position().x(), p.position().y(), p.position().z());
    }
    glEnd();

    pangolin::FinishFrame();
    usleep(5000);
    // sleep 5 ms
  }
}

}  // namespace proto_recon