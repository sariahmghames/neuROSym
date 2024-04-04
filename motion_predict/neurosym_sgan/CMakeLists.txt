cmake_minimum_required(VERSION 2.8.3)
project(neurosym_sgan)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries dependencies for this package.
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  people_msgs
  std_msgs
)



catkin_python_setup()


################################################
## Declare ROS messages, services and actions ##
################################################


#add_message_files(
#    FILES
#)

generate_messages(DEPENDENCIES std_msgs people_msgs darko_perception_msgs)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## declares dependencies for packages that depend on this package
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS roscpp rospy )



###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 ${catkin_INCLUDE_DIRS} )




# ==============================================
# Install
# ==============================================
#install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


file(GLOB python_scripts_DIR "src/neuROSym/motion_predict/${PROJECT_NAME}/scripts/*.py" "src/neuROSym/motion_predict/${PROJECT_NAME}/scripts/sgan/*.py" "src/neuROSym/motion_predict/${PROJECT_NAME}/scripts/data/*.py")
catkin_install_python(PROGRAMS ${python_scripts_DIR}  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

