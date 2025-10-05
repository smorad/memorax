from setuptools import find_packages, setup

setup(
    name="memorax",
    version="0.1.0",
    author="Steven Morad",
    author_email="stevenmorad@gmail.com",
    description="Deep memory and sequence modeling in JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/smorad/memorax",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "jaxtyping",
        "optax",
        "equinox",
        "beartype",
        "tqdm",
        "datasets",
        "pillow",
        "wandb",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
