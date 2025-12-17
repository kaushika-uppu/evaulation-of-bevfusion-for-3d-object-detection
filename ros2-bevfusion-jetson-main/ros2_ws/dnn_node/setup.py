from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dnn_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # This copies everything (onnx, engine, txt) from the models folder
        (os.path.join('lib', package_name, 'models'), glob('dnn_node/models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'classifier = dnn_node.classifier:main',
            'bev_node = dnn_node.bev_node:main',
            'replay_node = dnn_node.replay_node:main',
        ],
    },
)
