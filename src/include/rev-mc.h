/**
 * @file rev-mc.h
 * @brief Memory Controller Reverse Engineering Function Declarations
 * 
 * This header defines the core functions for reverse engineering DRAM
 * memory controller mappings through timing attacks and bitflip analysis.
 */

#include "utils.h"
#include "unistd.h"

/**
 * @brief Address tuple structure linking virtual and physical addresses
 * 
 * Associates a virtual address with its corresponding physical address,
 * essential for analyzing DRAM bank and row mappings.
 */
typedef struct {
	char* 		v_addr;  // Virtual address pointer
	uint64_t 	p_addr;  // Corresponding physical address
} addr_tuple;

//----------------------------------------------------------
// 			Function declarations

/**
 * @brief Performs timing measurements to analyze DRAM row buffer conflicts
 * 
 * Executes a series of memory accesses with precise timing to detect
 * row buffer conflicts, which reveal DRAM organization details.
 * 
 * @param rounds Number of timing rounds per measurement
 * @param m_size Size of memory region to analyze
 * @param o_file Output file path for results
 * @param flags Control flags for memory allocation
 * @param measurements Total number of measurements to perform
 */
void timing_measurement(size_t rounds, size_t m_size, char* o_file, uint64_t flags, size_t measurements);

/**
 * @brief Times access to a pair of memory addresses
 * 
 * Measures the access time to the second address after accessing the first,
 * allowing detection of row buffer conflicts and cache interactions.
 * 
 * @param a1 First address to access (row opener)
 * @param a2 Second address to time (victim)
 * @param rounds Number of timing rounds to average
 * @return Median access time in CPU cycles
 */
uint64_t time_tuple(volatile char* a1, volatile char* a2, size_t rounds);

/**
 * @brief Probes DRAM bank mappings using bitflip analysis
 * 
 * Systematically flips bits in physical addresses to determine which
 * bits control bank selection in the memory controller.
 * 
 * @param m_size Size of memory region to analyze
 * @param rounds Number of timing rounds per probe
 * @param flags Control flags for memory allocation
 * @param num_anchors Number of anchor addresses to test
 */
void mapping_bitflip_probe(size_t m_size, size_t rounds, size_t flags, int num_anchors);
