import unittest
from ngdlm.models.helpers import append_to_filepath
import os


class TestMisc(unittest.TestCase):

    def test_append_to_filepath(self):

        original_path = os.path.join("folder1", "folder1", "filename.ext")
        target_path = os.path.join("folder1", "folder1", "filename-modified.ext")

        transformed_path = append_to_filepath(original_path, "-modified")

        assert transformed_path == target_path


    def test_append_to_filepath_no_extension(self):

        original_path = os.path.join("folder1", "folder1", "filename")
        target_path = os.path.join("folder1", "folder1", "filename-modified")

        transformed_path = append_to_filepath(original_path, "-modified")

        assert transformed_path == target_path

if __name__ == '__main__':
    unittest.main()
