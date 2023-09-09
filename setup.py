from setuptools import setup, find_packages

# Read the README.md for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Read the requirements from requirements_prod.txt
with open("requirements_prod.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='lohrasb',
    version='4.2.0',
    author='drhosseinjavedani',
    author_email='h.javedani@gmail.com',
    description=("This versatile tool streamlines hyperparameter optimization in machine learning workflows."
                 "It supports a wide range of search methods, from GridSearchCV and RandomizedSearchCV"
                 "to advanced techniques like OptunaSearchCV, Ray Tune, and Scikit-Learn Tune."
                 "Designed to enhance model performance and efficiency, it's suitable for tasks of any scale."),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    license='BSD-3-Clause license',
    packages=find_packages(exclude=["examples*"]),
    include_package_data=True,  # This will read MANIFEST.in
    keywords=["Auto ML", "Pipeline", "Machine learning"],
    install_requires=requirements  # Use the parsed requirements here
)
