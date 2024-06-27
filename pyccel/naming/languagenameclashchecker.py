# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Superclass for handling name clash problems.
"""
from pyccel.utilities.metaclasses import Singleton
from pyccel.utilities.strings import create_incremented_string

class LanguageNameClashChecker(metaclass = Singleton):
    """
    Class containing functions to help avoid problematic names in a target language.

    A super class which provides functionalities to check or propose variable names and
    verify that they do not cause name clashes. Name clashes may be due to
    a variety of reasons which vary from language to language.

    Parameters
    ----------
    name : str
        The name of the class.
    bases : tuple[class,...]
        A tuple of the superclasses of the class.
    dct : dict
        A dictionary of the class attributes.
    """
    keywords = None

    def _get_collisionless_name(self, name, symbols):
        """
        Get a name which doesn't collision with keywords or symbols.

        Find a new name based on the suggested name which does not collision
        with the language keywords or the provided symbols.

        Parameters
        ----------
        name : str
            The suggested name.
        symbols : set
            Symbols which should be considered as collisions.

        Returns
        -------
        str
            A new name which is collision free.
        """
        if self.has_clash(name, symbols): #pylint: disable=no-member
            coll_symbols = self.keywords.copy()
            coll_symbols.update(symbols)
            counter = 1
            name, counter = create_incremented_string(coll_symbols,
                    prefix = name, counter = counter, name_clash_checker = self)
        return name
