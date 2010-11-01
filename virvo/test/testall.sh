#!/bin/bash
VV_SHADER_PATH=../shader gdb --args ./vview -testsuitefilename testall.csv "$@" -benchmark
