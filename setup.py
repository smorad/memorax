from setuptools import find_packages, setup

setup(
    name="memax",
    version="0.1.1",
    author="Steven Morad",
    author_email="stevenmorad@gmail.com",
    description="Deep memory and sequence modeling in JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/smorad/memax",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "jaxtyping",
        "optax",
        "beartype",
    ],
    extras_require={
        'equinox': ['equinox'],
        # TODO: Update if flax fixes their shit
        'flax': [
            'flax',
            'please-downgrade-to-python-3.13-for-flax; python_version >= "3.14"', 
        ],
        'train': [
            'datasets',
            'tqdm',
            'pillow',
            'wandb',
        ],
        'all': [
            'equinox',
            'flax',
            'please-downgrade-to-python-3.13-for-flax; python_version >= "3.14"',
            # train
            'datasets',
            'tqdm',
            'pillow',
            'wandb',
        ],
        'test': [
            'pytest',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
