#!/bin/bash
swig -c++ -python cbinder/CTree.i
g++ -fPIC -fopenmp -c cbinder/CTree_utils.cpp cbinder/CTree.cpp cbinder/CTree_wrap.cxx -I$1
g++ -fPIC -fopenmp -shared CTree_utils.o CTree.o CTree_wrap.o -o _CTree.so
rm CTree_utils.o CTree.o CTree_wrap.o
