# Makefile for DRAM Reverse Engineering Tool

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -g -Isrc/include -Ienable_arm_pmu
LDFLAGS = -lrt -lpthread

# Source files
SRCS = src/main.cpp src/utils.cpp src/rev-mc.cpp src/full_analysis.cpp src/linalg.cpp src/threshold.cpp
OBJS = $(SRCS:.cpp=.o)

# Tests
TEST_SRCS = tests/threshold/test_threshold.cpp
TEST_TARGETS = tests/bin/test_threshold

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

$(TEST_TARGETS): $(TEST_SRCS) src/threshold.cpp src/include/threshold.h
	mkdir -p tests/bin
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) src/threshold.cpp -o $@ $(LDFLAGS)

# Rule to compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET) $(TEST_TARGETS)

.PHONY: all clean test
