# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for qa4sm_reader.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
import subprocess
import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def run_transcribe_before_tests():
    script_path = "transcribe_before_testing.py"
    try:
        subprocess.run(["python", script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        pytest.fail(f"Failed to run {script_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        pytest.fail(f"Unexpected error running {script_path}")
