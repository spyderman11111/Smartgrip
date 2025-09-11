source /home/MA_SmartGrip/Smartgrip/py310/bin/activate

export SETUPTOOLS_USE_DISTUTILS=stdlib
python -m colcon build --packages-select gripanything --symlink-install

ros2 run gripanything seeanything_demo
