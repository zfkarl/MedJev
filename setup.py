from setuptools import find_packages, setup

setup(
    name='medjev',
    version='1.0.0',
    description='A system one vision-language decision model for medical image understanding',
    packages=find_packages(include=['medjev', 'medjev.*']),
    python_requires='>=3.10',
    install_requires=[line.strip() for line in open('requirements.txt') if line.strip()],
)
