from setuptools import setup, find_packages

# more info: https://docs.python.org/3/distutils/setupscript.html
# https://docs.pytest.org/en/latest/goodpractices.html#choosing-a-test-layout-import-rules

setup(
    name="NBSDynamics",
    packages=(find_packages(exclude=("tests"))),
    version="1.0.0",
    description="""
    A biophysical model framework written in Python to make simulations on coral development based on four environmental factors:
    (1) light; (2) flow; (3) temperature; and (4) acidity.
    For the hydrodynamics, the model can be coupled to Delft3D Flexible Mesh; a hydrodynamic model developed at Deltares """,
    license="LICENSE",
    long_description=open("README.md", encoding="utf8").read(),
    author="Jasper Dijkstra",
    author_email="jasper.dijkstra@deltares.nl",
    url="https://www.deltares.nl/nl/",
    download_url="https://github.com/Deltares/NBSDynamics",
    install_requires=[
        "netCDF4==1.4.2",
        "numpy==1.14.3",
        "scipy==1.1.0",
        "matplotlib==2.2.2",
    ],
)