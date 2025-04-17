#!/bin/bash

#g++ -std=c++11 -static-libstdc++ generate_test.cpp common.cpp -mavx512f -mavx512bw -O3 -o ./build/generate_test -lpthread

#g++ -std=c++11 -static-libstdc++ load_from_file_test.cpp common.cpp -mavx512f -mavx512bw -O3 -o ./build/load_from_file -lpthread

#g++ -std=c++11 -static-libstdc++ save_to_db_test.cpp common.cpp -mavx512f -mavx512bw -o ./build/save_to_db -lpthread -I ../../leveldb/include -L ../../leveldb/build -lleveldb

#g++ -std=c++11 -static-libstdc++ save_to_db_merge_test.cpp common.cpp -mavx512f -mavx512bw -O3 -o ./build/save_to_db_merge_test -lpthread -I ../../leveldb/include -L ../../leveldb/build -lleveldb

g++ -std=c++11 -static-libstdc++ save_to_file_test.cpp common.cpp -mavx512f -mavx512bw -o ./build/save_to_file_test -lpthread
