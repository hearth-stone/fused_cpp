CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -mcpu=native -Wall -Wextra -Wshadow -Wconversion -Wundef

TARGET := softmax_exp_constants
SOURCE := softmax_exp_constants.cpp

.PHONY: all clean assembly

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $@ $<

assembly: $(SOURCE)
	$(CXX) $(CXXFLAGS) -fno-asynchronous-unwind-tables -S -o softmax_exp_constants.s $<

clean:
	rm -f $(TARGET) softmax_exp_constants.s
