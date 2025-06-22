import setuptools as setup, find_packages

setup(
    name="babybert",
    version="0.0.0",
    description="Minimal BERT implementation in PyTorch",
    author="Drew Ross",
    author_email="drewross@ku.edu",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ]
)