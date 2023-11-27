from setuptools import find_packages,setup
from typing import List


def read_requirememts(file_path:str)->List[str]:

    requirements=[]
    hyphen="-e ."
    with open(file_path) as files:
        requirements=files.readlines()
        requirements=[requirement.replace("\n","") for requirement in requirements]
        if hyphen in requirements:
            requirements.remove("-e .")
    return requirements

setup(
name="mlproject",
version="0.0.1",
author_email="werayco@gmail.com",
packages=find_packages(),
install_requires=read_requirememts("requirements.txt")
)

