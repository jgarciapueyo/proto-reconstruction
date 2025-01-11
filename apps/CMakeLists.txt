include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(main_vo main_vo.cpp)
target_link_libraries(main_vo proto_reconstruction ${THIRD_PARTY_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(lab2 lab2.cpp)
target_link_libraries(lab2 proto_reconstruction ${THIRD_PARTY_LIBS})