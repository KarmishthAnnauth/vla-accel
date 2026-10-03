import os
from glob import glob

from setuptools import find_packages, setup

package_name = "recogdrive_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tools"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="recogdrive_ros maintainer",
    maintainer_email="user@example.com",
    description="ROS 2 interface for ReCogDrive VLA inference.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "recogdrive_node = recogdrive_ros.recogdrive_node:main",
            "image_decompress_node = recogdrive_ros.image_decompress_node:main",
            "stanley_controller_node = recogdrive_ros.stanley_controller_node:main",
        ],
    },
)
