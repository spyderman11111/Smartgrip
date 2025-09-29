source /home/MA_SmartGrip/Smartgrip/py310/bin/activate
source install/setup.bash 
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
  -p pose_frame:=base_link \ros
  -p hover_above:=0.30 \
  -p yaw_deg:=0.0 \
  -p ik_timeout:=2.0 \
  -p move_time:=3.0

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

^C(py310) MA_SmartGrip@Artemis:~/Smartgrip/ros2_ws$ ros2 run gripanything seeanything
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 24818.37it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 9039.45it/s]
[INFO] [1759154443.857214517] [seeanything_minimal_clean]: [seeanything_minimal_clean] topic=/my_camera/pylon_ros2_camera_node/image_raw, hover=0.400m, bias=(-0.050,-0.170,0.000); N=8, R=0.050m, orient_with_path=True
[INFO] [1759154443.925562448] [seeanything_minimal_clean]: 已发布关节初始位姿（T=5.0s）…
[INFO] [1759154443.940369104] [seeanything_minimal_clean]: 已收到 /joint_states（6 关节）。
[INFO] [1759154449.257247080] [seeanything_minimal_clean]: 初始位姿到位，开始等待检测。
/home/MA_SmartGrip/Smartgrip/py310/lib/python3.10/site-packages/transformers/models/grounding_dino/processing_grounding_dino.py:99: FutureWarning: The key `labels` is will return integer ids in `GroundingDinoProcessor.post_process_grounded_object_detection` output since v4.51.0. Use `text_labels` instead to retrieve string object names.
  warnings.warn(self.message, FutureWarning)
[INFO] [1759154450.086452361] [seeanything_minimal_clean]: [detect once] C_corr=(0.435,0.530,0.000), hover_z=0.400
[INFO] [1759154450.135026933] [seeanything_minimal_clean]: IK 服务可用：/compute_ik
[INFO] [1759154450.144461563] [seeanything_minimal_clean]: 已发布关节目标：[0.687720, -1.126022, 1.244994, -1.689768, -1.570796, -0.883076], T=3.0s
[INFO] [1759154453.458169818] [seeanything_minimal_clean]: 顶点序列生成：9 个。
[INFO] [1759154453.509205341] [seeanything_minimal_clean]: 已发布关节目标：[0.642898, -1.041521, 1.117445, -1.646720, -1.570796, 0.642898], T=5.0s
[INFO] [1759154458.857555942] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154459.159328420] [seeanything_minimal_clean]: 已发布关节目标：[0.694470, -0.995474, 1.045172, -1.620495, -1.570796, 1.479868], T=5.0s
[INFO] [1759154464.507456798] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154464.759664083] [seeanything_minimal_clean]: 已发布关节目标：[0.742285, -1.023920, 1.090048, -1.636925, -1.570796, 2.313082], T=5.0s
[INFO] [1759154470.107476307] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154470.359208537] [seeanything_minimal_clean]: 已发布关节目标：[0.761633, -1.109138, 1.220050, -1.681708, -1.570796, 3.117828], T=5.0s
[INFO] [1759154475.707512234] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154476.008984273] [seeanything_minimal_clean]: 已发布关节目标：[0.737527, -1.201574, 1.353248, -1.722470, -1.570796, 3.879120], T=5.0s
[INFO] [1759154481.357601781] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154481.659206989] [seeanything_minimal_clean]: 已发布关节目标：[0.679874, -1.247855, 1.416763, -1.739704, -1.570796, 4.606865], T=5.0s
[INFO] [1759154487.007517115] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154487.309369163] [seeanything_minimal_clean]: 已发布关节目标：[0.627069, -1.219254, 1.377769, -1.729311, -1.570796, 5.339458], T=5.0s
[INFO] [1759154492.657462497] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154492.910358800] [seeanything_minimal_clean]: 已发布关节目标：[0.613798, -1.133764, 1.256343, -1.693375, -1.570796, 6.111586], T=5.0s
[INFO] [1759154498.257477847] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154498.517592768] [seeanything_minimal_clean]: 已发布关节目标：[-2.125135, -2.100071, -1.117446, 4.788313, 1.570796, -5.266728], T=5.0s
[INFO] [1759154503.857473416] [seeanything_minimal_clean]: 到位，停顿拍照…
[INFO] [1759154504.157794574] [seeanything_minimal_clean]: 已发布关节初始位姿（T=5.0s）…
[INFO] [1759154510.007715802] [seeanything_minimal_clean]: seeanything_minimal_clean 圆周完成并返回初始姿态，退出。