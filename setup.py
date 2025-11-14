from setuptools import setup, find_packages
import os

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

def read_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='dimensio',
    version='0.2.0',
    author='Lingching Tung',
    author_email='lingchingtung@stu.pku.edu.cn',
    description='A flexible configuration space compression library for Bayesian Optimization',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/Elubrazione/dimensio',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.990',
        ],
    },
    keywords='bayesian-optimization hyperparameter-tuning configuration-space compression machine-learning auto-ml',
    project_urls={
        'Bug Reports': 'https://github.com/Elubrazione/dimensio/issues',
        'Source': 'https://github.com/Elubrazione/dimensio',
        'Documentation': 'https://github.com/Elubrazione/dimensio#readme',
    },
)