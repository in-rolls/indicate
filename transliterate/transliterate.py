#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

from keras.models import load_model
from pkg_resources import resource_filename
from .utils import download_file, REPO_BASE_URL


class Transliterate(object):
    MODELFN = None
    INPUT_VOCAB = None
    TARGET_VOCAB = None

    @classmethod
    def load_model_data(cls, latest=False):
        model = None
        input_vocab = None
        target_vocab = None
        failed = False

        if cls.MODELFN:
            model_fn = resource_filename(__name__, cls.MODELFN)
            path = os.path.dirname(model_fn)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(model_fn) or latest:
                print("Downloading model data from the server ({0!s})...".format(model_fn))
                if not download_file(REPO_BASE_URL + cls.MODELFN, model_fn):
                    print("ERROR: Cannot download model data file")
                    failed = True
            else:
                print("Using model data from {0!s}...".format(model_fn))

        if cls.INPUT_VOCAB:
            input_vocab_file = resource_filename(__name__, cls.INPUT_VOCAB)
            path = os.path.dirname(input_vocab_file)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(input_vocab_file) or latest:
                print("Downloading model data from the server ({0!s})...".format(input_vocab_file))
                if not download_file(REPO_BASE_URL + cls.INPUT_VOCAB, input_vocab_file):
                    print("ERROR: Cannot download input vocabulary file")
                    failed = True
            else:
                print("Using input vocab data from {0!s}...".format(input_vocab_file))

        if cls.TARGET_VOCAB:
            output_vocab_file = resource_filename(__name__, cls.TARGET_VOCAB)
            path = os.path.dirname(output_vocab_file)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(output_vocab_file) or latest:
                print("Downloading model data from the server ({0!s})...".format(output_vocab_file))
                if not download_file(REPO_BASE_URL + cls.TARGET_VOCAB, output_vocab_file):
                    print("ERROR: Cannot download target vocabulary file")
                    failed = True
            else:
                print("Using output vocab data from {0!s}...".format(output_vocab_file))

        if not failed:
            print("Loading model ...")
            model = load_model(model_fn)
            print("Loading input vocabulary ...")
            with open(input_vocab_file, "r") as fh:
                input_vocab = json.load(fh)
            print("Loading input vocabulary ...")
            with open(output_vocab_file, "r") as fh:
                target_vocab = json.load(fh)

        return model, input_vocab, target_vocab
