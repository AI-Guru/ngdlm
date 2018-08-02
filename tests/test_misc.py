import unittest
from ngdlm import models as ngdlmodels
from keras import models, layers
import logging
import numpy as np
import sys
import os


class TestMisc(unittest.TestCase):

    def test_append_to_filepath(self):

        original_path = os.path.join("folder1", "folder1", "filename.ext")
        target_path = os.path.join("folder1", "folder1", "filename-modified.ext")

        transformed_path = ngdlmodels.append_to_filepath(original_path, "-modified")

        assert transformed_path == target_path


    def test_append_to_filepath_no_extension(self):

        original_path = os.path.join("folder1", "folder1", "filename")
        target_path = os.path.join("folder1", "folder1", "filename-modified")

        transformed_path = ngdlmodels.append_to_filepath(original_path, "-modified")

        assert transformed_path == target_path

if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "SomeTest.testSomething" ).setLevel( logging.DEBUG )
    unittest.main()
