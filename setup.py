from setuptools import setup, find_packages

setup(
    name='JEST',
    version='0.1.0',
    description="jplace evaluation and scoring tool",
    author='Ali Alabiad',
    author_email="ali.z03@yahoo.com"
    url="https://github.com/AZA-2003/JEST",
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
    'JEST': ['JEST/_CTree.so'],
    },
    zip_safe=False,
    install_requires=[]
)
