# Makefile for DRAM Reverse Engineering Tool

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -g -Wall -Wextra -MMD -MP -Isrc/include -Ienable_arm_pmu
LDFLAGS = -lrt -lpthread

# Source files
SRCS = src/main.cpp src/utils.cpp src/rev-mc.cpp src/full_analysis.cpp src/linalg.cpp src/threshold.cpp
OBJS = $(SRCS:.cpp=.o)

# Tests
TEST_TARGETS = tests/bin/test_threshold tests/bin/test_linalg

# Target executable
TARGET = main

# Default target
all: $(TARGET)

test: $(TEST_TARGETS)
	@passed=0; failed=0; total=0; \
	for t in $(TEST_TARGETS); do \
		total=$$((total + 1)); \
		printf "[TEST] %s\n" "$$t"; \
		"$$t"; \
		status=$$?; \
		if [ $$status -eq 0 ]; then \
			passed=$$((passed + 1)); \
			printf "  PASS %s\n" "$$t"; \
		else \
			failed=$$((failed + 1)); \
			printf "  FAIL %s (exit %d)\n" "$$t" "$$status"; \
		fi; \
	done; \
	printf "\nTest summary: %d passed | %d failed | %d total\n" $$passed $$failed $$total; \
	if [ $$failed -ne 0 ]; then exit 1; fi

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

tests/bin/test_threshold: tests/threshold/test_threshold.cpp src/threshold.cpp src/include/threshold.h
	mkdir -p tests/bin
	$(CXX) $(CXXFLAGS) tests/threshold/test_threshold.cpp src/threshold.cpp -o $@ $(LDFLAGS)

tests/bin/test_linalg: tests/linalg/test_linalg.cpp src/linalg.cpp src/include/linalg.h
	mkdir -p tests/bin
	$(CXX) $(CXXFLAGS) tests/linalg/test_linalg.cpp src/linalg.cpp -o $@ $(LDFLAGS)

# Rule to compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(TARGET) $(TEST_TARGETS)

.PHONY: all clean test

-include $(OBJS:.o=.d)
