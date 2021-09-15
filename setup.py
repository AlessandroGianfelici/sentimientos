from setuptools import setup, find_packages

setup(name='sentimientos',
      version='0.1.0',
      description='Sentiment polarity tool for Italian',
      url='https://github.com/AlessandroGianfelici/sentimientos',
      author='Alessandro Gianfelici',
      author_email='alessandro.gianfelici@hotmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'tensorflow', 'spacy', 'numpy',
      ],
      zip_safe=False)