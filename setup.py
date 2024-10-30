from setuptools import find_packages, setup
setup(
    name="DeepLensingFlow",
    version="0.0",
    author= "Joaquin Armijo",
    author_email = 'joaquin.armijo@ipmu.jp',
    description="NF and DM image training framework",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["torch"],
    package_data={"NormalizingFlow": ["flows.py"]},
)

