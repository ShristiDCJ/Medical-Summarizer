from setuptools import find_packages, setup

setup(
    name="transformer-summarizer",
    version="0.1.0",
    description="Custom transformer model for medical transcript summarization",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.3",
        "torch>=2.5.1",
        "tokenizers>=0.20.1",
        "rouge-score>=0.1.2",
        "tqdm>=4.66.5",
        "tensorboard>=2.17.1",
    ],
    python_requires=">=3.11",
)

