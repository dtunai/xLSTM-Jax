from setuptools import setup, find_packages

with open("setup-requirements.txt", "r") as req_file:
    install_requires = req_file.read().splitlines()

setup(
    name="xlstm_jax",
    version="0.1.0",
    author="Dogukan Uraz Tuna",
    author_email="dogukanutuna@gmail.com",
    description="Packaged version of xLSTM architecture for Jax/Flax",
    url="https://github.com/dtunai/xlstm-jax",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
