from setuptools import setup, find_packages

setup(name='badgyal',
      version='0.0.1',
      description='pytorch badgyal and meangirl inference engine',
      author='dkappe',
      url='https://github.com/dkappe/badgyal',
      packages=find_packages(),
      package_data={'badgyal': ['*.pb.gz']},
      install_requires=[
          'torch',
          'numpy',
          'python-chess',
          'protobuf',
          'pylru'
      ]
)
