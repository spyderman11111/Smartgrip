source /home/MA_SmartGrip/Smartgrip/py310/bin/activate
source install/setup.bash 
ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py

export SETUPTOOLS_USE_DISTUTILS=stdlib

ros2 control list_controllers

source /opt/ros/humble/setup.bash
source ~/Smartgrip/py310/bin/activate
cd ~/Smartgrip/ros2_ws

rm -rf build/gripanything install/gripanything log

python3 -m colcon build --packages-select gripanything --symlink-install
source install/setup.bash
head -n 1 install/gripanything/lib/gripanything/seeanything

python3 -m gripanything.seeanything

ros2 control switch_controllers \
  --activate scaled_joint_trajectory_controller \
  --deactivate joint_trajectory_controller

ros2 run gripanything seeanything



source /home/sz/Smartgrip/.aria/bin/activate
# 2. 把示例代码解压到 Smartgrip 目录
python -m aria.extract_sdk_samples --output /home/sz/Smartgrip

# 3. 进入示例代码目录
cd /home/sz/Smartgrip/projectaria_client_sdk_samples

# 4. 安装依赖（建议先激活你的 venv 再执行）
python3 -m pip install -r requirements.txt


python gaze_stream.py --interface usb --update_iptables


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

ros2 run gripanything publish_object_points_tf \
  --ros-args \
  -p object_json:=/home/MA_SmartGrip/Smartgrip/ros2_ws/pc_out_simple/object_in_base_link.json \
  -p parent_frame:=base_link \
  -p prefix:=obj \
  -p report_tool0:=true \
  -p report_target:=center

ros2 run gripanything goto_point_from_object_json \
  --ros-args \
  -p object_json:=/home/MA_SmartGrip/Smartgrip/ros2_ws/pc_out_simple/object_in_base_link.json \
  -p use_point:=center \
  -p z_offset:=0.70
