import unittest
from ngdlm import models as ngdlmodels
from keras import models, layers
import logging
import numpy as np
import sys
    

if __name__ == '__main__':
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "SomeTest.testSomething" ).setLevel( logging.DEBUG )
    unittest.main()
