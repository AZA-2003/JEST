from setuptools import setup, find_packages, Distribution


class BinaryDistribution(Distribution):
    # The extension is built by compile_CTree.sh; wheels must still carry an
    # interpreter/platform tag rather than advertise this binary as pure Python.
    def has_ext_modules(self):
        return True


setup(
    name='JEST-Phylo',
    distclass=BinaryDistribution,
    version='0.3.2',
    description="jplace evaluation and scoring tool",
    author='Ali Alabiad',
    author_email="ali.z03@yahoo.com",
    url="https://github.com/AZA-2003/JEST",
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
    'JEST': ['_CTree.so'],
    },
    zip_safe=False,
    install_requires=[],
    entry_points={
        'console_scripts': ["JEST=JEST.JEST:app",]
                }
)
