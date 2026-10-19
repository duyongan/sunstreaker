from setuptools import setup, find_packages

__version__ = "2.0.0"

setup(
    name='sunstreaker',
    version=__version__,
    packages=find_packages(exclude=["examples"]),
    python_requires=">=3.12",
    install_requires=[
        "matplotlib>=3.9",
        "msgpack>=1.0.5",
        "networkx>=3.2",
        "pandas>=2.2",
        "dill>=0.3.8",
        "tqdm>=4.66",
        "numpy>=2.2",
    ],
    url='https://github.com/duyongan/sunstreaker',
    license='',
    author='duyongan (杜永安)【公众号：无数据不智能】',
    author_email='13261051171@163.com',
    description='源码清晰明了，使用简单好搞',
    classifiers=[
        "Programming Language :: Python :: 3.10"
    ]
)
