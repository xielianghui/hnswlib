#!/bin/bash

g++ -std=c++11 generate_test.cpp common.cpp -o generate_test -I ../../leveldb/include -lleveldb -L ../../leveldb/build