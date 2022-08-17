try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

torch_min = "1.11.0"
install_requires = [">=".join(["torch", torch_min]), "scikit-learn", "scipy", "gpytorch", "botorch", "dash", "dash_bootstrap_components"]
setup(
    name="alse",
    version="0.0.1",
    packages=["alse"],
    install_requires=install_requires,
)
