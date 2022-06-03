from setuptools import setup

long_description = open('README.md').read()

setup(
    name="shroom",
    version='0.2.0',
    description="Automated difference detection between data sets",
    long_description = long_description,
    long_description_content_type='text/markdown',
    author="Peter Melchior",
    author_email="peter.m.melchior@gmail.com",
    license='MIT',
    py_modules=["shroom"],
    url="https://github.com/astro-data-lab/shroom",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    install_requires=["numpy", "scipy", "matplotlib", "minisom", "corner"]
)
