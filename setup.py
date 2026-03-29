from setuptools import setup, find_packages

setup(
    name="turboquant-search",
    version="0.2.0",
    author="Tarun",
    description="Vector compression for similarity search — 6-10x compression, 84-92% recall, zero training.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tarun-ks/turboquant_search",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "demo": ["gradio>=4.0.0", "matplotlib>=3.7.0"],
        "faiss": ["faiss-cpu>=1.7.0"],
        "all": ["gradio>=4.0.0", "matplotlib>=3.7.0", "faiss-cpu>=1.7.0", "datasets>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "tqs=turboquant_search.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
