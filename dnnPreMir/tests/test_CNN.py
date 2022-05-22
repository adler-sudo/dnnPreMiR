#!/usr/bin/env

import pytest

import tensorflow as tf

from dnnPreMir.CNN import (
    CNNModel,
    CNNTrain,
    CNNEvaluation,
    CNNMain
)


# TODO: come up with ways to assert the model exists
# TODO: not sure best way to make this happen
def test_CNN_model():
    model = CNNModel.CNN_model()

def test_CNN_train():
    pass

if __name__ == "__main__":
    test_CNN_model()
    test_CNN_train()
