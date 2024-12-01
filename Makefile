ROOT_DIR= $(shell pwd)
TARGETS= toolkits/bc toolkits/bfs toolkits/cc toolkits/pagerank toolkits/sssp
# TARGETS= gim_toolkits/bc gim_toolkits/bfs gim_toolkits/cc gim_toolkits/pagerank toolkits/sssp
# MACROS= -DCXL_SHM
# MACROS= -D PRINT_DEBUG_MESSAGES

MPICXX= $(OpenMPI_DIR)/bin/mpicxx
CXXFLAGS= -O3 -Wall -std=c++11 -g -fopenmp -march=native -I$(ROOT_DIR)  $(MACROS)
SYSLIBS= -lnuma
HEADERS= $(shell find . -name '*.hpp')

all: $(TARGETS)

toolkits/%: toolkits/%.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

clean: 
	rm -f $(TARGETS)

