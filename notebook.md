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

 ros2 run tf2_ros tf2_echo base_link tool0

mid
At time 1757685244.63774557
- Translation: [0.243, 0.430, 0.270]
- Rotation: in Quaternion (xyzw) [0.999, 0.045, -0.016, -0.005]
- Rotation: in RPY (radian) [-3.130, 0.032, 0.090]
- Rotation: in RPY (degree) [-179.361, 1.839, 5.129]
- Matrix:
  0.995  0.089 -0.033  0.243
  0.089 -0.996  0.008  0.430
 -0.032 -0.011 -0.999  0.270
  0.000  0.000  0.000  1.000


right top 
At time 1757685323.349766970
- Translation: [0.437, 0.589, 0.270]
- Rotation: in Quaternion (xyzw) [0.999, 0.045, -0.016, -0.005]
- Rotation: in RPY (radian) [-3.130, 0.032, 0.091]
- Rotation: in RPY (degree) [-179.317, 1.854, 5.220]
- Matrix:
  0.995  0.091 -0.033  0.437
  0.091 -0.996  0.009  0.589
 -0.032 -0.012 -0.999  0.270
  0.000  0.000  0.000  1.000

right down
  At time 1757685393.59774548
- Translation: [0.300, 0.127, 0.270]
- Rotation: in Quaternion (xyzw) [0.999, 0.045, -0.014, -0.006]
- Rotation: in RPY (radian) [-3.128, 0.027, 0.090]
- Rotation: in RPY (degree) [-179.223, 1.567, 5.158]
- Matrix:
  0.996  0.090 -0.028  0.300
  0.090 -0.996  0.011  0.127
 -0.027 -0.014 -1.000  0.270
  0.000  0.000  0.000  1.000

left top 
At time 1757685444.663770236
- Translation: [-0.009, 0.568, 0.270]
- Rotation: in Quaternion (xyzw) [0.999, 0.045, -0.017, -0.003]
- Rotation: in RPY (radian) [-3.134, 0.034, 0.090]
- Rotation: in RPY (degree) [-179.561, 1.928, 5.138]
- Matrix:
  0.995  0.089 -0.034 -0.009
  0.090 -0.996  0.005  0.568
 -0.034 -0.008 -0.999  0.270
  0.000  0.000  0.000  1.000

  left down
At time 1757685516.51764067
- Translation: [0.118, 0.356, 0.269]
- Rotation: in Quaternion (xyzw) [0.999, 0.044, -0.018, 0.021]
- Rotation: in RPY (radian) [3.101, 0.037, 0.088]
- Rotation: in RPY (degree) [177.693, 2.115, 5.051]
- Matrix:
  0.995  0.089 -0.033  0.118
  0.088 -0.995 -0.043  0.356
 -0.037  0.040 -0.999  0.269
  0.000  0.000  0.000  1.000


seeanything
MA_SmartGrip@Artemis:~/Smartgrip/ros2_ws$ ros2 run tf2_ros tf2_echo base_link object_position
[INFO] [1757685975.621224115] [tf2_echo]: Waiting for transform base_link ->  object_position: Invalid frame ID "base_link" passed to canTransform argument target_frame - frame does not exist
At time 1757685977.334263379
- Translation: [0.141, 0.345, 0.000]
- Rotation: in Quaternion (xyzw) [0.000, 0.000, 0.000, 1.000]
- Rotation: in RPY (radian) [0.000, -0.000, 0.000]
- Rotation: in RPY (degree) [0.000, -0.000, 0.000]
- Matrix:
  1.000  0.000  0.000  0.141
  0.000  1.000  0.000  0.345
  0.000  0.000  1.000  0.000
  0.000  0.000  0.000  1.000


demo
At time 1757686713.996442420
- Translation: [1.183, -1.666, 0.000]
- Rotation: in Quaternion (xyzw) [0.000, 0.000, 0.000, 1.000]
- Rotation: in RPY (radian) [0.000, -0.000, 0.000]
- Rotation: in RPY (degree) [0.000, -0.000, 0.000]
- Matrix:
  1.000  0.000  0.000  1.183
  0.000  1.000  0.000 -1.666
  0.000  0.000  1.000  0.000
  0.000  0.000  0.000  1.000


(py310) MA_SmartGrip@Artemis:~/Smartgrip/ros2_ws$ ros2 run gripanything goto_p_top --ros-args -p preview:=true -p execute:=false
[INFO] [1757685622.242276595] [goto_p_top_via_ik]: 节点已启动: 订阅 /p_top, 参考系=base_link, 组=ur_manipulator, eef=tool0, move_time=3.0s, 避碰=False, preview=True, execute=False
[INFO] [1757685622.262522335] [goto_p_top_via_ik]: 收到 /p_top: p=(0.141,0.345,0.500), q=(1.000,0.000,0.000,0.000)
[INFO] [1757685622.262897378] [goto_p_top_via_ik]: 等待 /compute_ik 服务...
[WARN] [1757685623.265713904] [goto_p_top_via_ik]: 尚未收到 /joint_states，使用零位作为 IK 种子，解可能欠佳
[ERROR] [1757685625.266985519] [goto_p_top_via_ik]: IK 调用未返回或失败