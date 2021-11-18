"""
Copyright (C) Stichting Deltares 2021. All rights reserved.

This file is part of the SingleRunner.

The SingleRunner is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.
"""

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import List

import pytest

try:
    from pip import main as pipmain
except Exception as e_info:
    from pip._internal import main as pipmain


class TestUtils:

    _name_external = "externals"
    _name_local = "test_data"
    _name_artifacts = "artifacts"
    _temp_copies = "temp-copies"

    @staticmethod
    def install_package(package: str):
        """Installs a package that is normally only used
        by a test configuration.

        Arguments:
            package {str} -- Name of the PIP package.
        """
        pipmain(["install", package])

    @staticmethod
    def get_local_test_data_dir(dir_name: str) -> Path:
        """
        Returns the desired directory relative to the test data.
        Avoiding extra code on the tests.
        """
        directory = TestUtils.get_test_data_dir(dir_name, TestUtils._name_local)
        return directory

    @staticmethod
    def get_external_repo(dir_name: str) -> Path:
        """
        Returns the parent directory of this repo directory.

        Args:
            dir_name (str): Repo 'sibbling' of the current one.

        Returns:
            Path: Path to the sibbling repo.
        """
        return Path(__file__).parent.parent.parent / dir_name

    @staticmethod
    def get_test_data_dir(dir_name: str, test_data_name: str) -> Path:
        """
        Returns the desired directory relative to the test external data.
        Avoiding extra code on the tests.
        """
        return Path(__file__).parent / test_data_name / dir_name

    @staticmethod
    def get_local_test_file(filepath: str) -> Path:
        return Path(__file__).parent / TestUtils._name_local / filepath

    @staticmethod
    @contextlib.contextmanager
    def working_directory(path: Path):
        """Changes working directory and returns to previous on exit."""
        prev_cwd = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)
