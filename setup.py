from setuptools import setup, find_packages

setup(
      name = "mlkit",
      version = "0.1",
      packages = find_packages(),
      install_requires=[
          "numpy"
      ],
      description="A simple and modular machine learning library inspired by sklearn",
      author = "Ishaan A",
      url = "https://github.com/IshaanA-projects/mlkit",
      classifiers=[
          "Programming language :: Python :: 3",
          "Operating System:: OS Independent"
      ],
      
)