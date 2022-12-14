from setuptools import setup, find_packages

__version__ = "0.0.2.dev.1"

setup(
    name='sunstreaker',
    version=__version__,
    packages=find_packages(exclude=["examples"]),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib==3.6.2",
        "msgpack==1.0.4",
        "networkx==2.8.8",
        "pandas==1.5.1",
        "dill==0.3.6",
        "tqdm==4.64.1",
        "numpy==1.23.4",
        "msgpack==1.0.4"
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
