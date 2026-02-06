source /home/MA_SmartGrip/Smartgrip/aria/bin/activate
cd /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples

需要修改源码/home/MA_SmartGrip/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/inference/model/model_utils.py
line 203 model_buffer = torch.load(chkpt_path, map_location=map_location, weights_only=False)

python /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/stream_rgb_eye_sam2.py \
  --interface usb \
  --calib-vrs /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
  --output-dir /home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage


python /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/stream_rgb_eye_sam2.py \
  --interface wifi \
  --device-ip 192.168.0.102 \
  --calib-vrs /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
  --update_iptables \
  --output-dir /home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage




source /home/MA_SmartGrip/Smartgrip/py310/bin/activate
source install/setup.bash 
ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py

export SETUPTOOLS_USE_DISTUTILS=stdlib

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


