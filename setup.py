from setuptools import setup, find_packages

setup(
    name='pca_sphere_projection',
    version='0.1.0',
    description='A visualization and analysis tool for PCA projections on a unit sphere',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Long Yuan',
    author_email='lyuan13@jhmi.edu',
    url='https://github.com/imlong4real/pca_sphere_projection',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'statsmodels',
        'plotly',
        'streamlit'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
