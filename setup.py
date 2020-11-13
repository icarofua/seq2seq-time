from setuptools import setup, find_packages

setup(
    name='seq2seq_time',
    packages=find_packages(),
    version='0.1.0',
    description='Using sequence to sequence interfaces for timeseries regression',
    author='Icaro Oliveira',
    author_email='icarofua@gmail.com',
    license='MIT',
    install_requires=[
    'performer-pytorch==0.9',
    'reformer-pytorch==1.2.3',
    'linear-attention-transformer==0.15.0',
    'routing_transformer==1.4.1',
    'memformer==0.3.0',
    'sinkhorn-transformer==0.11.1',
    ],
)

