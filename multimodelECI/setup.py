try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

torch_min = "1.9"
install_requires = [">=".join(["torch", torch_min]), "scikit-learn", "scipy"]
setup(
    name="multimodelECI",
    version="0.0.1",
    packages=["helper"],
    install_requires=install_requires,
)