include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(proto_reconstruction SHARED
        features/feature_matcher.cpp
        geometry/pose_estimation.cpp
        utils/imagestream.cpp
        visualization/mapvisualizer.cpp
        vo/frame.cpp
        vo/map.cpp
        vo/mappoint.cpp
        vo/vo.cpp
)
target_link_libraries(proto_reconstruction ${THIRD_PARTY_LIBS})
