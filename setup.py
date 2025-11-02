from setuptools import setup

package_name = 'alpp_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[], # We are not installing this as a library
    py_modules=[
    'Pure_Pursuit.ALPP',
    'Pure_Pursuit.mock_odom_publisher'  # <-- ADD THIS LINE
],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vatshvan',
    maintainer_email='your_email@example.com',
    description='Adaptive Lookahead Pure Pursuit Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'alpp_node = Pure_Pursuit.ALPP:main',
            'mock_odom = Pure_Pursuit.mock_odom_publisher:main',  # <-- ADD THIS LINE
        ],
    },
)
