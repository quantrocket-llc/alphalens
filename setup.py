#!/usr/bin/env python
from setuptools import setup, find_packages
import versioneer

if __name__ == "__main__":
    setup(
        name='alphalens',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Performance analysis of predictive (alpha) stock factors',
        author='Quantopian Inc.',
        author_email='opensource@quantopian.com',
        packages=find_packages(include='alphalens.*'),
        package_data={
            'alphalens': ['examples/*'],
        },
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python',
            'Topic :: Utilities',
            'Topic :: Office/Business :: Financial',
            'Topic :: Scientific/Engineering :: Information Analysis',
        ],
        url='https://github.com/quantopian/alphalens'
    )
