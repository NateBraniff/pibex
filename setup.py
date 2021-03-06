#from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pibex", 
    version="0.0.1",
    author='Nate Braniff',
    author_email='nate.braniff@gmail.com',
    description="A package for optimal Bayesian GLM response surface designs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NateBraniff/pibex",
    keywords = ['Bayesian','Response Surface','Optimization','Optimal','Experimental','Design'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6.1',
    install_requires = [
        'numpy>=1.17.0',
        'pandas>=1.0.0',
        'matplotlib',
        'patsy>=0.5.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license = 'LGPLv3+'
    # py_modules=["model","design"],
    # package_dir={'':'nloed'},
)