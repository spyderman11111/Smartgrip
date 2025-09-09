ros2 run image_view image_saver \
  --ros-args \
  -r image:=/my_camera/pylon_ros2_camera_node/image_raw \
  -r camera_info:=/my_camera/pylon_ros2_camera_node/camera_info \
  -p save_all_image:=false \
  -p filename_format:="/home/MA_SmartGrip/snapshot.jpg"
