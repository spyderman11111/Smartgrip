from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'gripanything'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MA_SmartGrip',
    maintainer_email='shuo.zhang@campus.tu-berlin.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seeanything = gripanything.seeanything:main',
            'seeanything_demo = gripanything.seeanything_demo:main',
            'goto_hover_once = gripanything.goto_hover_once:main',
            'detect_and_circle_scan_vggt = gripanything.detect_and_circle_scan_vggt:main',
        ],
    },
)
