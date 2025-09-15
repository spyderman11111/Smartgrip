source /home/MA_SmartGrip/Smartgrip/py310/bin/activate

ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py

export SETUPTOOLS_USE_DISTUTILS=stdlib
python -m colcon build --packages-select gripanything --symlink-install

ros2 run gripanything seeanything_demo

(u,v)=(1009.8,585.3)  C_base=(0.335,0.655,0.000)  P_top=(0.335,0.655,0.500)  


[INFO] [1757686567.062916547] [seeanything_minimal]: uv=(2005.8,730.6)  object=(1.182,-1.664,0.000)
[INFO] [1757686567.253325676] [seeanything_minimal]: 检测出3个 yellow object
[INFO] [1757686567.455981534] [seeanything_minimal]: uv=(2005.8,731.0)  object=(1.183,-1.666,0.000)
[INFO] [1757686567.638205444] [seeanything_minimal]: 检测出3个 yellow object
[INFO] [1757686567.840890413] [seeanything_minimal]: uv=(89.0,123.5)  object=(-0.090,-0.347,-0.000)
[INFO] [1757686568.025339994] [seeanything_minimal]: 检测出3个 yellow object


ros2 run gripanything goto_hover_once --ros-args \
  -p object_frame:=object_position \
  -p pose_frame:=base_link \
  -p hover_above:=0.30 \
  -p yaw_deg:=0.0 \
  -p ik_timeout:=2.0 \
  -p move_time:=3.0

---
header:
  stamp:
    sec: 1757949080
    nanosec: 989252914
  frame_id: base_link
name:
- shoulder_lift_joint
- elbow_joint
- wrist_1_joint
- wrist_2_joint
- wrist_3_joint
- shoulder_pan_joint
position:
- -1.186562405233719
- 1.1997712294207972
- -1.5745235882201136
- -1.5696094671832483
- -0.579871956502096
- 0.9239029288291931
effort:
- -3.7297627925872803
- -2.2732200622558594
- -0.4983561635017395
- 0.018187813460826874
- -0.005302319303154945
- -0.0032134205102920532
---