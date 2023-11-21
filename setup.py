from setuptools import find_packages,setup
from typing import List

def get_requirements(file:str)-> List[str]:
    requirements=[]
    with open(file) as file_path:
        requirements=file_path.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
name="mlproject",
version="0.0.1",
author_email="werayco@gmail.com"
packages=find_packages(where='srs')
install_requires=get_requirements("requirements.txt")
)