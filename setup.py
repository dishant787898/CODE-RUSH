from setuptools import setup, find_packages

setup(
    name="waste-classification-app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.0.1",
        "tensorflow>=2.9.1",
        "numpy>=1.23.1",
        "Pillow>=9.2.0",
        "python-dotenv>=0.20.0",
        "gunicorn>=20.1.0",
        "requests>=2.28.1",
        "tqdm>=4.64.1"
    ],
    python_requires=">=3.8,<3.10",
)
