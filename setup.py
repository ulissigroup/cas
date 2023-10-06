try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

torch_min = "1.11.0"
install_requires = [
    ">=".join(["torch", torch_min]),
    "ipympl",
    "scikit-learn",
    "scipy",
    "openpyxl",
    "gpytorch",
    "botorch",
    "dash",
    "dash_bootstrap_components",
]
setup(
    name="cas",
    version="0.0.1",
    packages=["cas"],
    install_requires=install_requires,
)
