source /home/MA_SmartGrip/Smartgrip/aria/bin/activate

USB：python stream_rgb_eye_sam2.py --interface usb

WiFi：python stream_rgb_eye_sam2.py --interface wifi --device-ip 192.168.0.102

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
ros2 run gripanything seeanything_debug


source /home/sz/Smartgrip/.aria/bin/activate
# 2. 把示例代码解压到 Smartgrip 目录
python -m aria.extract_sdk_samples --output /home/sz/Smartgrip

# 3. 进入示例代码目录
cd /home/sz/Smartgrip/projectaria_client_sdk_samples

# 4. 安装依赖（建议先激活你的 venv 再执行）
python3 -m pip install -r requirements.txt


python gaze_stream.py --interface usb --update_iptables


python3 -m pip install projectaria_client_sdk --no-cache-dir

ros2 run gripanything publish_object_points_tf \
  --ros-args \
  -p object_json:=/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/offline_output/object_in_base_link.json \
  -p parent_frame:=base_link \
  -p prefix:=obj \
  -p report_tool0:=true \
  -p report_target:=center

ros2 run gripanything goto_point_from_object_json \
  --ros-args \
  -p object_json:=/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/offline_output/object_in_base_link.json\
  -p use_point:=center \
  -p z_offset:=0.20
