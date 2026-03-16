from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    with open(file_path) as f:
        requirements = f.read().splitlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements
    


setup(
    name='mlproject',
    version='0.1.0',
    author='Urvil Patel',
    author_email='uvpatel7271@gmail.com',
    packages=find_packages(),
    # install_requires=[
    #     'numpy',
    #     'pandas',
    #     'seaborn',
    #     'matplotlib',
    #     'scikit-learn',
    #     'xgboost',
        
    # ]
    install_requires=get_requirements('requirements.txt')
)