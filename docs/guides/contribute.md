# How to contribute to NBSDynamics
<!-- #### Table of Contents:
1. [Tooling](#tooling)
    1. [Development environment](#dev_env)
    2. [Code convention / Linters](#code_convention)
    3. [Continuous Integration](#continuous_integration)
    4. [Testing](#testing)
    5. [Version control](#version_control)
    6. [Documentation](#documentation)
2. [Development](#development)
    1. [Branches](#branches)
    2. [Reviews](#reviews)
    3. [Merging](#merging)
    4. [Coding guidelines](#coding_guidelines) -->

## 1. Tooling
In this section we describe which tools this repository relays on to guarantee a standardized development.

### 1.1. Development environment.
In order to develop on this project it is recommended the usage of a virtual environment. This can be easily achieved by using poetry (see below).

#### Poetry
We use `poetry` to manage our package and its dependencies, which you can download [here](https://python-poetry.org/).
After installation, make sure it's available on your `PATH` and run it in the HYDROLIB-core directory in your shell of choice.

To install the package (by default in editable mode) run `poetry install`. We advise using `virtualenv`s, Poetry will create one for you.
If you need to use an already existing Python installation, you can activate it and run `poetry env use system` before `poetry install`.

#### Pip
Latest versions of `pip` support installing packages from a .toml file. This streamlines the process so that you do not need to use poetry (although we highly recommend it in order to add packages that are compliant with the rest of the repository).

You can easily develop your own features using the pip edit mode from the root of the NBSDynamics checked out repository:

```
pip install -e .
```

#### Known issues 
We found out that packages such as `netcdf4` or `pypiwin32` / `pywin32` might give problems. In case you run against said problems while installing the package we recommend installing them beforehand.

### 1.2. Code convention / Linters
This project uses both black and isort as an autoformatter. It is recommended following the rules defined in pyproject.toml to avoid conflicts before a merge.

#### 1.2.1. Black
We use `black` as an autoformatter. Black will curate the code follows the PEP8 convention. It is also run during CI and will fail if it's not formatted beforehand.

#### 1.2.2. Isort
We use `isort` as an autoformatter. Isort will curate the imports of each module are ordered. It is also run during CI and will fail if it's not formatted beforehand.

### 1.3. Continuous Integration
Each commit made on a branch of the repository gets analyzed with <a href="https://sonarcloud.io/summary/new_code?id=Deltares_NBSDynamics">Sonar Cloud</a>. Before merging, the following rules should be fulfilled regarding new code:
- Code coverage remains does not drop.
- No new bugs introduced.
- No new code smells introduced.
- No new vulnerabilities introduced.
- No new security hotspots introduced.
- No new duplications of code.

### 1.4. Testing
New code should be tested. As a rule of thumb public methods should be tested with unit tests and new workflows with integration tests. Acceptance tests are recommended when introducing new models.
We use `pytest` to test our package. Run it with `poetry run pytest` to test your code changes locally.

### 1.5. Version Control
We use [`commitizen`](https://commitizen-tools.github.io/commitizen/) to automatically bump the version number.
If you use [conventional commit messages](https://www.conventionalcommits.org/en/v1.0.0/#summary), the the [`changelog.md`](../changelog.md) is generated automatically.

### 1.6. Documentation
We use `mkdocs` to automatically generate documentation. We define documentation in separate sections:

* /guides: where we include information on how to use the repository, either as maintainer or as user.
* /reference: where the technical documentation is linked. When creating a new module a new markdown file should be created. To refer to a module, it can be done as follows:
    ```markdown
    ### core
    ::: src.core.core
    ```
    This will generate documentation based on the docstrings of each class and method in src.core.core.py
* changelog.md: file automatically generated and updated with the commits to master (see [Version Control](#version_control))

## 2. Development.
In this section we describe how development is expected to be done in this repository.

### 2.1. Branches
For each issue or feature, a separate branch should be created from the main. To keep the branches organized a feature branch should be created with the `feature/` prefix.
When starting development on a branch, a pull request should be created for reviews and continous integration. During continuous integration, the checks will be run with python 3.8 on Windows, Ubuntu and MacOS. The checks consist of running the tests, checking the code formatting and running SonarCloud. 
We advise to use a draft pull request, to prevent the branch to be merged back before developement is finished. When the branch is ready for review, you can update the status of the pull request to "ready for review".

### 2.2. Reviews
Reviews should be dona by a member of the development team on a pull-request prior to its merging.

### 2.3. Merging
Merging a branch can only happen when a pull request is accepted through review. When a pull request is accepted the changes should be merged back with the "squash and merge" option.

### 2.4. Coding guidelines
* Classes and methods should make use of docstrings.
* If there are any additions or changes to the public API, the documentation should be updated. 
* Files should be added to the appropriate folder to keep modules and objects within the correct scope. 
* If there is code that needs to be tested, there should be tests written for it.
    * Tests should be added "mirroring" the structure of src for a cohesive project layout.