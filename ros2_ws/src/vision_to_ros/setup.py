from setuptools import find_packages, setup

package_name = 'vision_to_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MA_SmartGrip',
    maintainer_email='shuo.zhang@campus.tu-berlin.de',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'smartgrip = vision_to_ros.smartgrip:main'
        ],
    },
)
