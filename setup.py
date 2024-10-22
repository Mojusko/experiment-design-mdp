from setuptools import setup

packages=['numpy',
          'autograd',
          'stpy',
          'torch',
          'pytest',
          'scipy',
          ]

setup(name='doexpy',
      version='0.0.2',
      description='',
      url='',
      author='Mojmir Mutny, Tadeusz Janik, Jose Pablo Folch',
      author_email='mojmir.mutny@inf.ethz.ch',
      license='MIT Licence',
      packages=['doexpy'],
	    zip_safe=False,
      install_requires=packages)