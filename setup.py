from setuptools import setup, find_packages

setup(
    name        = "rlcarla",
    version     = "0.2.0",
    description = "Custom RL Autonomous Driving Environment for CARLA 0.9.16",
    packages    = find_packages(),
    install_requires = [
        "gymnasium",
        "numpy",
        "torch",
        "pygame",
        "opencv-python",
        "tensorboard",
    ],
    python_requires = ">=3.10",
)
