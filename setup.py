from setuptools import setup, find_packages

setup(
    name="turboquant-search",
    version="0.1.0",
    author="Tarun",
    description="Vector compression for similarity search, inspired by TurboQuant (ICLR 2026)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tarun-ks/turboquant_search",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "demo": ["gradio>=4.0.0", "matplotlib>=3.7.0"],
        "faiss": ["faiss-cpu>=1.7.0"],
        "all": ["gradio>=4.0.0", "matplotlib>=3.7.0", "faiss-cpu>=1.7.0", "datasets>=2.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
