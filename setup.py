from setuptools import setup, find_packages

# Simple fixed version
VERSION = '0.1.0'

setup(
    name="waste-classification-app",
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9,<3.11",
    install_requires=[],  # We'll handle this in requirements.txt
)
        "requests>=2.28.1",
        "tqdm>=4.64.1"
    ],
    python_requires=">=3.8,<3.10",
)
