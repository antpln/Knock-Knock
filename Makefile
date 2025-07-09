SDIR=src
IDIR=$(SDIR)/include
LDIR=lib
BUILD=obj
ODIR=src/.obj

CXXFLAGS=-I$(IDIR) -O0 -g#-ggdb
CXX=g++
LDFLAGS=

OUT=tester

LDEPS=

GB_PAGE=/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
HUGEPAGE=/mnt/huge

all: $(OUT)
.PHONY: clean


SOURCES := $(wildcard $(SDIR)/*.cpp)
OBJECTS := $(patsubst $(SDIR)/%.cpp, $(ODIR)/%.o, $(SOURCES))

# Assembly sources
# ASM_SOURCES := $(wildcard $(SDIR)/*.S)
# ASM_OBJECTS := $(patsubst $(SDIR)/%.S, $(ODIR)/%.o, $(ASM_SOURCES))

$(ODIR)/%.o: $(SDIR)/%.cpp
	mkdir -p $(ODIR)
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(LDFLAGS) $(LDEPS)

$(ODIR)/%.o: $(SDIR)/%.S
	mkdir -p $(ODIR)
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(LDFLAGS) $(LDEPS)


$(OUT): $(OBJECTS) $(ASM_OBJECTS)
	mkdir -p $(BUILD)
	$(CXX) -o $(BUILD)/$@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDEPS)
	chmod +x $(BUILD)/$@

clean:
	rm -rf $(BUILD)
	rm -rf $(ODIR)
