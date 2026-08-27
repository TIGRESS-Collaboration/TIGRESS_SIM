cmake_minimum_required(VERSION 3.16)
project(GimpyBPureCylinderCo60Cone LANGUAGES CXX)

option(WITH_GEANT4_UIVIS "Build with Geant4 UI and visualization" ON)

if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(gimpy_b_co60_cone main.cc ${SOURCES} ${HEADERS})
target_compile_features(gimpy_b_co60_cone PRIVATE cxx_std_17)
target_link_libraries(gimpy_b_co60_cone ${Geant4_LIBRARIES})

configure_file(run.mac run.mac COPYONLY)
configure_file(vis.mac vis.mac COPYONLY)
