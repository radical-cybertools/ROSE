from setuptools import setup, find_packages

setup_args = {}

setup_args['name']                 = "rose"
setup_args['version']              = "0.1.0"
setup_args['packages']             = find_packages()
setup_args['python_requires']      = '>=3.6'
setup_args['install_requires']     = ['numpy', 'radical.asyncflow', 'pydantic']
setup(**setup_args)
