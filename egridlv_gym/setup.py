from setuptools import setup, find_packages

setup(
    name="egridlv_gym",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pandapower",
        "numpy",
        "matplotlib",
        "simbench==1.5.3",
    ],
)
