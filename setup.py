from setuptools import setup, find_packages

setup(
    name="intent_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "pandas>=1.3.0",
        "numpy>=1.19.5",
        "scikit-learn>=0.24.2",
        "optuna>=2.10.0",
        "openpyxl>=3.0.7",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An email intent classification system using RoBERTa",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
) 