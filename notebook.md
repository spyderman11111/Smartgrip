source /home/MA_SmartGrip/Smartgrip/py310/bin/activate
source install/setup.bash 
ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py

export SETUPTOOLS_USE_DISTUTILS=stdlib
python -m colcon build --packages-select gripanything --symlink-install

ros2 run gripanything seeanything



ros2 topic pub --once /scaled_joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
  ],
  points: [{
    positions: [0.9239029288291931, -1.186562405233719, 1.1997712294207972, -1.5745235882201136, -1.5696094671832483, -0.579871956502096],
    time_from_start: {sec: 3, nanosec: 0}
  }]
}"

- shoulder_lift_joint
- elbow_joint
- wrist_1_joint
- wrist_2_joint
- wrist_3_joint
- shoulder_pan_joint
position:
- -1.0457398456386109
- 1.0822847525226038
- -1.581707616845602
- -1.5601266066180628
- -0.8573678175555628
- 0.7734344005584717


python3 -m pip install projectaria_client_sdk --no-cache-dir