from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'My F110 learning results'
LONG_DESCRIPTION = 'DRL based f110 agents for autonomous racing'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="ff_racing", 
        version=VERSION,
        author="Benjamin Evans",
        author_email="<bdevans@sun.ac.za>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'autonomous racing'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux",
        ]
)
