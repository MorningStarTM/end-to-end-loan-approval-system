from setuptools import find_packages, setup
from typing import List

HYPEN = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as obj:
        requirements = obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN in requirements:
            requirements.remove(HYPEN)
    return requirements

setup(

    name='loan_approval_system',
    version='0.0.1',
    author="Ernest",
    author_email="ernestmachinelearning@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)