#!/usr/bin/env python

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from isPreMiR import (
    seq_process,
    parse_opt,
    second_struct_predict,
    transform_seq_struct,
    x_cast,
    predict_results
)


def test_seq_process():
    seq = "CUGA"
    seq_newline = "CUGA\n"
    seq_not_rna = "CTGA"
    seq_lower = "cuga"

    assert seq_process(seq) == "CUGA"
    assert seq_process(seq_newline) == "CUGA"
    assert seq_process(seq_lower) == "CUGA"
    
    with pytest.raises(SystemExit):
        seq_process(seq_not_rna)

def test_parse_opt():
    argv_seq_only = ["-s", "CUGA"]
    argv_all_short = ["-s", "CUGA", "-i", "infile.txt", "-o" "outfile.txt"]
    argv_all_long = ["--sequence", "CUGA", "--infile", "infile.txt", "--outfile", "outfile.txt"]
    argv_empty = []
    
    assert parse_opt(argv_seq_only) == ("CUGA","","")
    assert parse_opt(argv_all_short) == ("CUGA", "infile.txt", "outfile.txt")
    assert parse_opt(argv_all_long) == ("CUGA", "infile.txt", "outfile.txt")

    with pytest.raises(SystemExit):
        parse_opt(argv_empty)

# TODO: not sure how we are gonna test this one - this is placeholder solution for now
# depends on file being written to the ./temp/temp_sequence.fa
def test_second_struct_predict():
    seq = "CUGA"
    temp_sequence_file = "./temp/temp_sequence.fa"
    temp_seq_struct_file = "./temp/temp_seq_struct.fa"

    with open(temp_sequence_file, "w") as fd:
        fd.write(">\n")
        fd.write(seq)

    assert second_struct_predict(seq) == ["C.", "U.", "G.", "A."]

    with open(temp_seq_struct_file, "r") as f:
        arrow = f.readline().strip("\n")
        struct_seq = f.readline().strip("\n")
        character_seq = f.readline().strip("\n")

        assert arrow == ">"
        assert struct_seq == seq
        assert character_seq == ".... (  0.00)"

def test_transform_seq_struct():
    # TODO: add some additional options
    seq_struct = ["C."]
    seq_struct_empty = []
    
    assert transform_seq_struct(seq_struct, 1) == [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert len(transform_seq_struct(seq_struct)) == 180
    assert transform_seq_struct(seq_struct_empty) == [[0] * 12] * 180

def test_predict_results():
    # TODO: come up with additional test
    seq_struct_vector = np.array([[0] * 12] * 180).reshape(1, 180, 12)

    assert_array_equal(
        predict_results(seq_struct_vector),
        np.array([[0.57374895, 0.42625108]], dtype = np.float32)
    )

if __name__ == "__main__":
    test_seq_process()
    test_parse_opt()
    test_second_struct_predict()
    test_transform_seq_struct()
    test_predict_results()
