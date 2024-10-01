# This is something that build our Machine Learning Application as Package or Artifact

from typing import List
from setuptools import find_packages, setup

def get_packages(file_path:str)->List[str]:
    '''
    This Function is Responsible for getting the list of packages and installing it
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name='ML Project',
    version='0.0.1',
    author='Afaf Ahmed Khan',
    author_email='afafahmedkhan@gmail.com',
    packages=find_packages(),
    install_requires = get_packages('requirements.txt')
)