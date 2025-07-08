from setuptools import setup, find_packages

setup(
    name="egridlv_gym",
    version="0.1.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "gymnasium",
        "pandas",
        "numpy",
        "matplotlib",
        "pandapower",
        "simbench==1.5.3",
    ],
)
