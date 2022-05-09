from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

# with open('HISTORY.md') as history_file:
#     HISTORY = history_file.read()

setup(
    name="pubmad",
    version="0.26",
    author='Jacopo Bandoni, Pier Paolo Tarasco, William Simoni, Marco Natali',
    author_email="bandoni.jacopo@gmail.com",
    description="Useful tools to work with biology",
    long_description=README + '\n\n',
    long_description_content_type="text/markdown",
    url="https://github.com/Pier297/ProgettoBIO",
    packages=['pubmad'],
    python_requires=">=3.6",
    install_requires = [
    'biopython',
    'certifi',
    'charset-normalizer',
    'click',
    'cycler',
    'filelock',
    'fonttools',
    'huggingface-hub',
    'idna',
    'joblib',
    'kiwisolver',
    'matplotlib',
    'networkx',
    'nltk',
    'numpy',
    'packaging',
    'Pillow',
    'pymed',
    'pyparsing',
    'python-dateutil',
    'PyYAML',
    'regex',
    'requests',
    'sacremoses',
    'six',
    'tokenizers',
    'torch',
    'tqdm',
    'transformers',
    'typing-extensions',
    'urllib3',
    'pyvis'
]
)


    # 'biopython==1.79',
    # 'certifi==2021.10.8',
    # 'charset-normalizer==2.0.12',
    # 'click==8.1.2',
    # 'cycler==0.11.0',
    # 'filelock==3.6.0',
    # 'fonttools==4.33.2',
    # 'huggingface-hub==0.5.1',
    # 'idna==3.3',
    # 'joblib==1.1.0',
    # 'kiwisolver==1.4.2',
    # 'matplotlib==3.5.1',
    # 'networkx==2.8',
    # 'nltk==3.7',
    # 'numpy==1.22.3',
    # 'packaging==21.3',
    # 'Pillow==9.1.0',
    # 'pymed==0.8.9',
    # 'pyparsing==3.0.8',
    # 'python-dateutil==2.8.2',
    # 'PyYAML==6.0',
    # 'regex==2022.4.24',
    # 'requests==2.27.1',
    # 'sacremoses==0.0.49',
    # 'six==1.16.0',
    # 'tokenizers==0.12.1',
    # 'torch==1.11.0',
    # 'tqdm==4.64.0',
    # 'transformers==4.18.0',
    # 'typing-extensions==4.2.0',
    # 'urllib3==1.26.9',