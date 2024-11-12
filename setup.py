from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="Abaqus Seq2Seq flow curve calibration project",
    version="0.1.0",
    author="Nguyen Xuan Binh",
    author_email="binh.nguyen@aalto.fi",
    description="A project for ABAQUS hardening flow curve calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpringNuance/Abaqus-Hardening-Seq-2-Seq-Project",
    packages=find_packages(include=['src', 'src.*']),
    include_package_data=True,
    install_requires=[
        "numpy>=1.18.5",
        "pandas>=1.0.5",
        "scikit-learn>=0.23.1",
        "mlflow>=1.10.0",
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'my_ml_project=src.main_pipeline:main_pipeline',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
