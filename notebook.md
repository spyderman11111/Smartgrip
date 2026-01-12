source /home/MA_SmartGrip/Smartgrip/py310/bin/activate
source install/setup.bash 
ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py

export SETUPTOOLS_USE_DISTUTILS=stdlib
python -m colcon build --packages-select gripanything --symlink-install

ros2 control list_controllers

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

(py310) MA_SmartGrip@Artemis:~/Smartgrip/ros2_ws$ ros2 run gripanything seeanything
Enter the target text prompt (e.g., 'orange object'): yellow cube .
[INFO] [1760622642.436804465] [seeanything_minimal_clean]: Using user prompt: "yellow cube ."
[INFO] [1760622644.788373070] [seeanything_minimal_clean]: [seeanything] topic=/my_camera/pylon_ros2_camera_node/image_raw, hover=0.300m, bias=(-0.030,-0.230,0.000); N=30, R=0.100m, orient=radial_in, dir=ccw, sweep=355.0°
[INFO] [1760622644.883726490] [seeanything_minimal_clean]: Received /joint_states (6 joints).
[INFO] [1760622644.884408536] [seeanything_minimal_clean]: Published INIT joint pose (T=5.0s).
[INFO] [1760622650.188809425] [seeanything_minimal_clean]: INIT reached. Waiting for Stage-1 detection (XY only)…
[INFO] [1760622651.041911468] [seeanything_minimal_clean]: [detect once] score=0.87, C=(0.389,0.466,0.000), hover_z=0.300
[INFO] [1760622651.043363631] [seeanything_minimal_clean]: [Stage-1] Move XY→(0.389,0.466), keep Z=0.414
[INFO] [1760622651.101908505] [seeanything_minimal_clean]: IK service available: /compute_ik
[INFO] [1760622651.118604630] [seeanything_minimal_clean]: Published joint goal: [0.652859, -1.321121, 1.477196, -1.726871, -1.570796, -0.917938], T=3.0s
[INFO] [1760622654.439102132] [seeanything_minimal_clean]: Stage-1 done. Waiting for Stage-2 detection (XY + descend)…
[INFO] [1760622655.024638673] [seeanything_minimal_clean]: [detect once] score=0.88, C=(0.399,0.439,0.000), hover_z=0.300
[INFO] [1760622655.025177248] [seeanything_minimal_clean]: [Stage-2] Move XY→(0.399,0.439), Z→0.300
[INFO] [1760622655.033476875] [seeanything_minimal_clean]: Published joint goal: [0.606127, -1.272722, 1.718664, -2.016738, -1.570796, -0.964669], T=3.0s
[INFO] [1760622658.340978871] [seeanything_minimal_clean]: Generated vertices: 30 -> trimmed to 29 for sweep 355.0° / 360.0°.
[INFO] [1760622658.391009164] [seeanything_minimal_clean]: Published joint goal: [0.733103, -1.101549, 1.471921, -1.941168, -1.570796, -0.837743], T=3.0s
[INFO] [1760622661.739023141] [seeanything_minimal_clean]: At vertex 1, dwell for capture…
[INFO] [1760622661.874881566] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_1_image.png
[INFO] [1760622661.875574422] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_1" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622662.740956270] [seeanything_minimal_clean]: Published joint goal: [0.752245, -1.132783, 1.519115, -1.957128, -1.570796, -1.028040], T=3.0s
[INFO] [1760622666.089096565] [seeanything_minimal_clean]: At vertex 2, dwell for capture…
[INFO] [1760622666.225870855] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_2_image.png
[INFO] [1760622666.226727653] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_2" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622667.141064550] [seeanything_minimal_clean]: Published joint goal: [0.766426, -1.169667, 1.573640, -1.974769, -1.570796, -1.223298], T=3.0s
[INFO] [1760622670.489169343] [seeanything_minimal_clean]: At vertex 3, dwell for capture…
[INFO] [1760622670.623918190] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_3_image.png
[INFO] [1760622670.624849679] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_3" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622671.540992451] [seeanything_minimal_clean]: Published joint goal: [0.774712, -1.210999, 1.633130, -1.992927, -1.570796, -1.424452], T=3.0s
[INFO] [1760622674.889081594] [seeanything_minimal_clean]: At vertex 4, dwell for capture…
[INFO] [1760622675.026089199] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_4_image.png
[INFO] [1760622675.026946107] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_4" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622675.891072080] [seeanything_minimal_clean]: Published joint goal: [0.776196, -1.255426, 1.695106, -2.010476, -1.570796, -1.632408], T=3.0s
[INFO] [1760622679.239048523] [seeanything_minimal_clean]: At vertex 5, dwell for capture…
[INFO] [1760622679.373831784] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_5_image.png
[INFO] [1760622679.374771903] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_5" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622680.241356569] [seeanything_minimal_clean]: Published joint goal: [0.770074, -1.301424, 1.757038, -2.026411, -1.570796, -1.847970], T=3.0s
[INFO] [1760622683.589124613] [seeanything_minimal_clean]: At vertex 6, dwell for capture…
[INFO] [1760622683.724196475] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_6_image.png
[INFO] [1760622683.725177764] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_6" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622684.591300465] [seeanything_minimal_clean]: Published joint goal: [0.755770, -1.347243, 1.816391, -2.039945, -1.570796, -2.071713], T=3.0s
[INFO] [1760622687.938995883] [seeanything_minimal_clean]: At vertex 7, dwell for capture…
[INFO] [1760622688.075592875] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_7_image.png
[INFO] [1760622688.076606834] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_7" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622688.941202123] [seeanything_minimal_clean]: Published joint goal: [0.733106, -1.390873, 1.870671, -2.050594, -1.570796, -2.303816], T=3.0s
[INFO] [1760622692.289018417] [seeanything_minimal_clean]: At vertex 8, dwell for capture…
[INFO] [1760622692.422050024] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_8_image.png
[INFO] [1760622692.423235565] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_8" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622693.291062814] [seeanything_minimal_clean]: Published joint goal: [0.702512, -1.430046, 1.917498, -2.058248, -1.570796, -2.543850], T=3.0s
[INFO] [1760622696.639070588] [seeanything_minimal_clean]: At vertex 9, dwell for capture…
[INFO] [1760622696.767102366] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_9_image.png
[INFO] [1760622696.768147256] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_9" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622697.641010622] [seeanything_minimal_clean]: Published joint goal: [0.665209, -1.462332, 1.954712, -2.063176, -1.570796, -2.790592], T=3.0s
[INFO] [1760622700.989074224] [seeanything_minimal_clean]: At vertex 10, dwell for capture…
[INFO] [1760622701.125032401] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_10_image.png
[INFO] [1760622701.126003190] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_10" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622701.991080487] [seeanything_minimal_clean]: Published joint goal: [0.623294, -1.485381, 1.980506, -2.065922, -1.570796, -3.041947], T=3.0s
[INFO] [1760622705.339081627] [seeanything_minimal_clean]: At vertex 11, dwell for capture…
[INFO] [1760622705.474283534] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_11_image.png
[INFO] [1760622705.475351164] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_11" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622706.391034939] [seeanything_minimal_clean]: Published joint goal: [0.579610, -1.497286, 1.993576, -2.067086, -1.570796, -3.295070], T=3.0s
[INFO] [1760622709.739006989] [seeanything_minimal_clean]: At vertex 12, dwell for capture…
[INFO] [1760622709.868665514] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_12_image.png
[INFO] [1760622709.869791614] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_12" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622710.790418713] [seeanything_minimal_clean]: Published joint goal: [0.537378, -1.496981, 1.993243, -2.067059, -1.570796, -3.546742], T=3.0s
[INFO] [1760622714.139102241] [seeanything_minimal_clean]: At vertex 13, dwell for capture…
[INFO] [1760622714.277815714] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_13_image.png
[INFO] [1760622714.279004975] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_13" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622715.191044759] [seeanything_minimal_clean]: Published joint goal: [0.499667, -1.484493, 1.979525, -2.065828, -1.570796, -3.793893], T=3.0s
[INFO] [1760622718.539035052] [seeanything_minimal_clean]: At vertex 14, dwell for capture…
[INFO] [1760622718.687299868] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_14_image.png
[INFO] [1760622718.688550509] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_14" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622719.541185814] [seeanything_minimal_clean]: Published joint goal: [0.468900, -1.460940, 1.953133, -2.062990, -1.570796, -4.034099], T=3.0s
[INFO] [1760622722.889148261] [seeanything_minimal_clean]: At vertex 15, dwell for capture…
[INFO] [1760622723.029837836] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_15_image.png
[INFO] [1760622723.031231639] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_15" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622723.940950176] [seeanything_minimal_clean]: Published joint goal: [0.446573, -1.428258, 1.915400, -2.057939, -1.570796, -4.265865], T=3.0s
[INFO] [1760622727.289111629] [seeanything_minimal_clean]: At vertex 16, dwell for capture…
[INFO] [1760622727.428854573] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_16_image.png
[INFO] [1760622727.430208965] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_16" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622728.341032814] [seeanything_minimal_clean]: Published joint goal: [0.433238, -1.388809, 1.868153, -2.050140, -1.570796, -4.488640], T=3.0s
[INFO] [1760622731.689012251] [seeanything_minimal_clean]: At vertex 17, dwell for capture…
[INFO] [1760622731.826750035] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_17_image.png
[INFO] [1760622731.828418660] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_17" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622732.691247326] [seeanything_minimal_clean]: Published joint goal: [0.428684, -1.345020, 1.813567, -2.039343, -1.570796, -4.702634], T=3.0s
[INFO] [1760622736.039072369] [seeanything_minimal_clean]: At vertex 18, dwell for capture…
[INFO] [1760622736.169927079] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_18_image.png
[INFO] [1760622736.171551723] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_18" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622737.041074603] [seeanything_minimal_clean]: Published joint goal: [0.432187, -1.299147, 1.754028, -2.025677, -1.570796, -4.908570], T=3.0s
[INFO] [1760622740.389021264] [seeanything_minimal_clean]: At vertex 19, dwell for capture…
[INFO] [1760622740.524724715] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_19_image.png
[INFO] [1760622740.526058337] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_19" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622741.391033320] [seeanything_minimal_clean]: Published joint goal: [0.442747, -1.253189, 1.692035, -2.009642, -1.570796, -5.107449], T=3.0s
[INFO] [1760622744.739092073] [seeanything_minimal_clean]: At vertex 20, dwell for capture…
[INFO] [1760622744.878372973] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_20_image.png
[INFO] [1760622744.879936637] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_20" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622745.790808569] [seeanything_minimal_clean]: Published joint goal: [0.459253, -1.208880, 1.630122, -1.992039, -1.570796, -5.300383], T=3.0s
[INFO] [1760622749.138970582] [seeanything_minimal_clean]: At vertex 21, dwell for capture…
[INFO] [1760622749.270053777] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_21_image.png
[INFO] [1760622749.271494910] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_21" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622750.141131333] [seeanything_minimal_clean]: Published joint goal: [0.480591, -1.167737, 1.570820, -1.973879, -1.570796, -5.488485], T=3.0s
[INFO] [1760622753.489096866] [seeanything_minimal_clean]: At vertex 22, dwell for capture…
[INFO] [1760622753.616432926] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_22_image.png
[INFO] [1760622753.617862169] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_22" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622754.541019677] [seeanything_minimal_clean]: Published joint goal: [0.505694, -1.131104, 1.516601, -1.956294, -1.570796, -5.672821], T=3.0s
[INFO] [1760622757.889081963] [seeanything_minimal_clean]: At vertex 23, dwell for capture…
[INFO] [1760622758.021276234] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_23_image.png
[INFO] [1760622758.022973628] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_23" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622758.940992346] [seeanything_minimal_clean]: Published joint goal: [0.533567, -1.100177, 1.469826, -1.940445, -1.570796, -5.854387], T=3.0s
[INFO] [1760622762.289087644] [seeanything_minimal_clean]: At vertex 24, dwell for capture…
[INFO] [1760622762.412620107] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_24_image.png
[INFO] [1760622762.414165000] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_24" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[INFO] [1760622763.340912449] [seeanything_minimal_clean]: Published joint goal: [0.563281, -1.076001, 1.432642, -1.927437, -1.570796, -6.034113], T=3.0s
[INFO] [1760622766.689021441] [seeanything_minimal_clean]: At vertex 25, dwell for capture…
[INFO] [1760622766.814622000] [seeanything_minimal_clean]: Saved snapshot: /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/ur5image/pose_25_image.png
[INFO] [1760622766.816230614] [seeanything_minimal_clean]: Saved positions & camera pose under key "image_25" -> /home/MA_SmartGrip/Smartgrip/ros2_ws/build/gripanything/gripanything/image_jointstates.json
[WARN] [1760622767.714691849] [seeanything_minimal_clean]: Abnormal joint jump: joint=wrist_1_joint, Δ=2.959 rad (> 2.200), ref=-1.927 -> target=-4.886. Skipped.
[WARN] [1760622767.715398475] [seeanything_minimal_clean]: Vertex IK skipped after abnormal jump. Moving to next vertex.
[WARN] [1760622767.748200042] [seeanything_minimal_clean]: Abnormal joint jump: joint=elbow_joint, Δ=2.826 rad (> 2.200), ref=1.433 -> target=-1.394. Skipped.
[WARN] [1760622767.748905708] [seeanything_minimal_clean]: Vertex IK skipped after abnormal jump. Moving to next vertex.
[WARN] [1760622767.797802646] [seeanything_minimal_clean]: Abnormal joint jump: joint=wrist_2_joint, Δ=3.142 rad (> 2.200), ref=-1.571 -> target=-4.712. Skipped.
[WARN] [1760622767.798540402] [seeanything_minimal_clean]: Vertex IK skipped after abnormal jump. Moving to next vertex.
[WARN] [1760622767.847728912] [seeanything_minimal_clean]: Abnormal joint jump: joint=elbow_joint, Δ=2.840 rad (> 2.200), ref=1.433 -> target=-1.408. Skipped.
[WARN] [1760622767.848426888] [seeanything_minimal_clean]: Vertex IK skipped after abnormal jump. Moving to next vertex.
[INFO] [1760622767.889149234] [seeanything_minimal_clean]: Published INIT joint pose (T=5.0s).
[INFO] [1760622773.239072605] [seeanything_minimal_clean]: Circle finished. Returned to INIT. Exiting.