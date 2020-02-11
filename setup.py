import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="path_explain",
    version="0.0.5",
    author="Pascal Sturmfels",
    author_email="psturm@cs.washingoton.edu",
    description="A package for explaining attributions and interactions in deep neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suinleelab/path_explain",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)