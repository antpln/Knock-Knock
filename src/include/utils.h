/**
 * @file utils.h
 * @brief Utility functions and macros for DRAM reverse engineering
 * 
 * This header provides essential utilities for memory manipulation, timing,
 * cache operations, and memory allocation used in DRAM analysis attacks.
 */

#pragma once 

#include <stdint.h>
#include <stdio.h>

// Bit manipulation and memory size conversion macros
#define BIT(x) (1ULL<<(x))              // Create bitmask with bit x set
#define KB(x) ((x)<<10ULL)              // Convert to kilobytes
#define MB(x) ((x)<<20ULL)              // Convert to megabytes  
#define GB(x) ((x)<<30ULL)              // Convert to gigabytes
#define CL_SHIFT 6                      // Cache line shift (64 bytes = 2^6)
#define CL_SIZE 64                      // Cache line size in bytes

// Control flags for memory allocation and operations
#define F_CLEAR 	0L                  // Clear all flags
#define F_VERBOSE 	BIT(1)              // Enable verbose logging
#define F_EXPORT 	BIT(1)              // Enable data export

// Memory allocation flags (using bits 30-34 for memory-related flags)
#define MEM_SHIFT			(30L)
#define MEM_MASK			0b11111ULL << MEM_SHIFT				
#define F_ALLOC_HUGE 		BIT(MEM_SHIFT)          // Use huge pages
#define F_ALLOC_HUGE_1G 	F_ALLOC_HUGE | BIT(MEM_SHIFT+1)  // Use 1GB huge pages
#define F_ALLOC_HUGE_2M		F_ALLOC_HUGE | BIT(MEM_SHIFT+2)  // Use 2MB huge pages
#define F_POPULATE			BIT(MEM_SHIFT+3)        // Populate pages immediately



//----------------------------------------------------------
// 			Static inline functions for cache and timing operations

/**
 * @brief Flush a cache line containing the given address (ARM64)
 * 
 * Uses the DC CIVAC instruction to clean and invalidate the cache line
 * containing the specified address, ensuring it's evicted from all cache levels.
 * 
 * @param p Pointer to the address whose cache line should be flushed
 */
static inline __attribute__((always_inline)) void clflush(volatile void *p) {
    asm volatile("dc civac, %0\n"     // Clean and invalidate by VA to PoC
                 "dsb ish\n"          // Data synchronization barrier
                 :
                 : "r"(p)
                 : "memory");
}

/**
 * @brief Memory fence - ensures all memory operations complete before proceeding
 * 
 * Uses DSB SY (Data Synchronization Barrier - System) to ensure all explicit
 * memory accesses that appear in program order before this instruction are
 * observed before any explicit memory accesses that appear after it.
 */
static inline __attribute__((always_inline)) void mfence(void) {
    asm volatile("dsb sy" ::: "memory");
}

/**
 * @brief Load fence - ensures all loads complete before proceeding
 * 
 * Uses DSB LD + ISB to ensure all load operations complete and that
 * the instruction pipeline is flushed before continuing execution.
 */
static inline __attribute__((always_inline)) void lfence(void) {
    asm volatile("dsb ld\n"           // Data synchronization barrier for loads
                 "isb\n"              // Instruction synchronization barrier
                 ::: "memory");
}


/**
 * @brief Read timestamp counter (ARM64 virtual timer)
 * 
 * Reads the system virtual counter (CNTVCT_EL0) which provides a consistent
 * time reference across cores. Used for high-precision timing measurements.
 * 
 * @return Current virtual timer value in ticks
 */
static inline __attribute__((always_inline)) uint64_t rdtsc(void) {
    uint64_t virtual_timer_value;
    asm volatile("isb\n"                // Prevent speculative execution reordering
                 "mrs %0, cntvct_el0\n" // Read system virtual counter
                 : "=r"(virtual_timer_value));
    return virtual_timer_value;
}

/**
 * @brief Read timestamp counter with serialization (ARM64)
 * 
 * Similar to rdtsc but includes additional barriers to ensure strict
 * ordering of memory operations and prevent timing measurement interference.
 * 
 * @return Current virtual timer value in ticks
 */
static inline __attribute__((always_inline)) uint64_t rdtscp(void) {
    uint64_t timer_value;
    asm volatile("isb\n"              // Instruction synchronization barrier
                 "mrs %0, cntvct_el0\n" // Read virtual counter
                 "dsb sy\n"           // Data synchronization barrier
                 : "=r"(timer_value));
    return timer_value;
}


//----------------------------------------------------------
// 			Memory allocation structures and functions

/**
 * @brief Memory buffer structure for managing allocated memory regions
 * 
 * Contains information about an allocated memory buffer including its
 * virtual address, size, and allocation flags used for DRAM analysis.
 */
typedef struct {
	char* 		buffer;    // Virtual address of the allocated buffer
	uint64_t 	size;      // Size of the buffer in bytes
	uint64_t 	flags;     // Allocation flags (huge pages, population, etc.)
} mem_buff_t;

/**
 * @brief Allocate a memory buffer with specified properties
 * 
 * Allocates memory with the properties specified in the mem_buff_t structure,
 * including support for huge pages and immediate population.
 * 
 * @param mem Pointer to mem_buff_t structure with allocation parameters
 * @return 0 on success, -1 on failure
 */
int alloc_buffer(mem_buff_t* mem);

/**
 * @brief Free a previously allocated memory buffer
 * 
 * Unmaps the memory region allocated by alloc_buffer().
 * 
 * @param mem Pointer to mem_buff_t structure containing buffer to free
 * @return 0 on success, -1 on failure
 */
int free_buffer(mem_buff_t* mem);

