#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Hindi to English translation
"""

import unittest

from indicate.hindi2english import HindiToEnglish

class TestHindiToEnglish(unittest.TestCase):
    def test_hindi_to_english(self):
        test_inputs = ["राजशेखर चिंतालपति","गौरव सूद"]
        test_outputs = ["rajshekhar chintalpati","gaurav sood"]
        for hindi, english in zip(test_inputs, test_outputs):
            self.assertEqual(HindiToEnglish.transliterate(hindi), english)

if __name__ == "__main__":
    unittest.main()
