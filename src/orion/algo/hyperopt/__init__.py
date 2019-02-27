# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.hyperopt -- TODO
======================================

.. module:: hyperopt
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
from ._version import get_versions

VERSIONS = get_versions()
del get_versions

__descr__ = 'TODO'
__version__ = VERSIONS['version']
__license__ = 'BSD 3-Clause'
__author__ = u'Mike Pieper'
__author_short__ = u'Mike Pieper'
__author_email__ = 'mpieper636@gmail.com'
__copyright__ = u'2019, Mike Pieper'
__url__ = 'https://github.com/mikepieper/orion.algo.hyperopt'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
