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


skiplinux = pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="Linux not supported"
)


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
    def get_artifact_testcase_copy(dir_src: Path, test_name: str) -> Path:
        copy_dir = TestUtils.get_artifact_test_data_dir(test_name)
        if copy_dir.is_dir():
            shutil.rmtree(copy_dir)
        # Path.mkdir(copy_dir, parents=True)
        try:
            shutil.copytree(dir_src, copy_dir)
        except Exception as e_info:
            print(f"Error copying tree {str(e_info)}")

        return copy_dir

    @staticmethod
    def get_temporary_folder() -> Path:
        import tempfile

        copy_dir: Path = Path(tempfile.mkdtemp())
        if copy_dir.is_dir():
            shutil.rmtree(copy_dir)
        return copy_dir

    @staticmethod
    def get_testcase_local_copy(dir_name: Path) -> Path:
        copy_dir: Path = TestUtils.get_temporary_folder()
        try:
            shutil.copytree(dir_name, copy_dir)
        except shutil.Error as e_info:
            raise Exception(f"Error copying tree {str(e_info)}") from e_info

        return copy_dir

    @staticmethod
    def get_local_test_data_dir(dir_name: str) -> Path:
        """
        Returns the desired directory relative to the test data.
        Avoiding extra code on the tests.
        """
        directory = TestUtils.get_test_data_dir(dir_name, TestUtils._name_local)
        return directory

    @staticmethod
    def get_external_test_data_dir(dir_name: str) -> Path:
        """
        Returns the desired directory relative to the test external data.
        Avoiding extra code on the tests.
        """
        test_dir = Path(__file__).parent
        return test_dir / TestUtils._name_external / dir_name

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
    def copy_test_dir_into_artifacts_dir(
        test_name: str,
        *results_dirs,
    ) -> Path:
        """Copies all the result directories that need to be moved
        into the artifacts directory for later analysis

        Args:
            test_name (str): Name for the new artifacts directory.
            results_dir: List of unpacked directories to move into the artifacts directory.
        Returns:
            Path: Path to generated artifact test dir.
        """
        artifacts_dir = TestUtils.get_artifact_test_data_dir(test_name)
        if artifacts_dir.is_dir():
            shutil.rmtree(artifacts_dir)
        list_dirs_to_copy: List[Path] = list(results_dirs)
        for result_dir in list_dirs_to_copy:
            if result_dir.exists():
                shutil.copytree(result_dir, artifacts_dir / result_dir.name)
        return artifacts_dir

    @staticmethod
    def get_external_system_test_data_dir(dir_name: str) -> Path:
        """
        Returns the desired directory relative to the system test data within
        the external test directory. Avoiding extra code on the tests.

        Args:
            dir_name (str): Subdirectory in system test data folder.

        Returns:
            Path: Valid test data path.
        """
        return TestUtils.get_external_test_data_dir("system_test_data") / dir_name

    @staticmethod
    def get_artifact_test_data_dir(dir_name: str) -> Path:
        """
        Returns the desired directory relative to the test artifacts (local copy)
        data. Avoiding extra code on the tests.
        """
        return TestUtils.get_test_data_dir(dir_name, TestUtils._name_artifacts)

    @staticmethod
    def get_test_data_dir(dir_name: str, test_data_name: str) -> Path:
        """
        Returns the desired directory relative to the test external data.
        Avoiding extra code on the tests.
        """
        return Path(__file__).parent / test_data_name / dir_name

    @staticmethod
    def get_test_dir(dir_name: str) -> Path:
        """Returns the desired directory inside the Tests folder

        Arguments:
            dir_name {str} -- Target directory.

        Returns:
            {str} -- Path to the target directory.
        """
        test_dir = Path(__file__).parent
        dir_path = test_dir / dir_name
        return dir_path

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
