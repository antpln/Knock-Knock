/**
 * @file rev-mc.cpp
 * @brief DRAM Memory Controller Reverse Engineering Implementation
 * 
 * This file implements the core algorithms for reverse engineering DRAM
 * memory controller mappings using timing-based side-channel attacks.
 * It includes both timing measurement and bitflip analysis techniques.
 */

#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <string.h>

#include <vector>
#include <algorithm>

#include "rev-mc.h"

#ifdef __aarch64__
/**
 * @brief Read ARM64 performance counter (cycle counter)
 * 
 * Reads the Performance Monitors Cycle Count Register (PMCCNTR_EL0)
 * which provides high-resolution cycle counting for timing measurements.
 * 
 * @return Current cycle count
 */
static inline uint64_t read_pmccntr(void)
{
    uint64_t val;
    asm volatile("mrs %0, pmccntr_el0" : "=r"(val));
    return val;
}
#endif



/**
 * @brief Measures access time between two memory addresses with high precision
 * 
 * This function implements the core timing attack for detecting DRAM row buffer
 * conflicts. It first accesses address a1 (row opener), then measures the time
 * to access a2 (victim). If both addresses are in the same DRAM row, a2 will
 * be faster (row buffer hit). If in different rows, a2 will be slower (row buffer miss).
 * 
 * The implementation includes several anti-speculation techniques:
 * - Cache line eviction to eliminate cache effects
 * - Memory barriers to prevent out-of-order execution
 * - Fake dependency creation to prevent speculative loads
 * 
 * @param a1 First address to access (row opener)
 * @param a2 Second address to measure timing for (victim)
 * @param rounds Number of measurements to perform for statistical reliability
 * @return Median access time in CPU cycles
 */
uint64_t time_tuple(volatile char *a1, volatile char *a2, size_t rounds)
{
    std::vector<uint64_t> times;
    times.reserve(rounds);
    uint64_t t0, t1;
    
    for (size_t i = 0; i < rounds; i++)
    {
        #ifdef __aarch64__
        // Evict both addresses from the cache to eliminate cache effects
        asm volatile("DC CIVAC, %0" ::"r"(a1));  // Clean & invalidate a1
        asm volatile("DC CIVAC, %0" ::"r"(a2));  // Clean & invalidate a2

        // Memory barriers to ensure cache operations complete
        asm volatile("DSB SY");  // Data synchronization barrier
        asm volatile("ISB");
        asm volatile("NOP; NOP; NOP; NOP; NOP; NOP; NOP; NOP; NOP; NOP");
        // Access a1 to open its row
        volatile char val = *(volatile char *)a1;

        // Conditional branch to prevent OoO speculation
        if (val == 0xAB)
        {
            asm volatile("nop" ::: "memory");
        }

        // Create a fake dependency on val, but don’t change the address
        // this allows to avoid speculative load of a2
        uintptr_t safe_ptr;
        asm volatile(
            "add %0, %1, #0\n\t"
            "eor %0, %0, %2\n\t"
            "eor %0, %0, %2\n\t"
            : "=r"(safe_ptr)
            : "r"(a2), "r"(val)
            : "memory");
        volatile char *safe_a2 = (char *)safe_ptr;
        assert(safe_a2 == a2 && "safe_a2 should equal a2 (dependency transformation failed!)");

        asm volatile("NOP; NOP; NOP; NOP; NOP; NOP; NOP; NOP; NOP; NOP");

        // Time access to actual a2
        asm volatile("DSB SY");
        asm volatile("ISB");
        asm volatile("mrs %0, pmccntr_el0" : "=r"(t0)); // Time start
        val = *(volatile char *)safe_a2;
        asm volatile("DSB SY");
        asm volatile("ISB");
        asm volatile("mrs %0, pmccntr_el0" : "=r"(t1)); // Time end

        times.push_back(t1 - t0);
        #else

        asm volatile("clflush (%0)" ::"r"(a1));
        asm volatile("clflush (%0)" ::"r"(a2));
        asm volatile("mfence"); // Ensure all cache flushes are complete
        asm volatile("lfence");
        
        volatile char val = *(volatile char *)a1;
        asm volatile("mfence"); // Ensure the load is complete before timing
        asm volatile("lfence");
        asm volatile("rdtsc" : "=A"(t0)); // Time start
        val = *(volatile char *)a2;
        asm volatile("mfence"); // Ensure the load is complete before timing
        asm volatile("lfence");
        asm volatile("rdtsc" : "=A"(t1)); // Time end

        times.push_back(t1 - t0);
        #endif
        

    }
    std::sort(times.begin(), times.end());
    uint64_t median_time = times[times.size() / 2];
    return median_time;
}

/**
 * @brief Generates a random address within the allocated memory buffer
 * 
 * Selects a random byte offset within the memory buffer to use for
 * timing measurements, ensuring uniform sampling across the address space.
 * 
 * @param base Base address of the memory buffer
 * @param m_size Size of the memory buffer in bytes
 * @return Random address within the buffer bounds
 */
char *get_rnd_addr(char *base, size_t m_size)
{
    uint64_t random_offset = rand() % m_size;

    // Ensure the resulting address does not exceed base + m_size
    if ((uint64_t)base + random_offset >= (uint64_t)base + m_size)
    {
        random_offset = m_size - 1; // Clamp to last byte
    }

    return (char *)((uint64_t)base + random_offset);
}

/**
 * @brief Structure for mapping physical frame numbers to virtual addresses
 * 
 * Used to maintain a mapping between physical page frame numbers (PFNs)
 * and their corresponding virtual addresses for efficient address translation.
 */
typedef struct
{
    uint64_t pfn;  // Physical Frame Number (page-aligned)
    uint64_t va;   // Virtual Address (page-aligned)
} pfn_va_t;

//----------------------------------------------------------
/**
 * @brief Extracts the Physical Frame Number from a pagemap entry
 * 
 * Extracts the PFN from a /proc/self/pagemap entry by masking out
 * the control bits, leaving only the frame number.
 * 
 * @param entry Raw pagemap entry from /proc/self/pagemap
 * @return Physical Frame Number
 */
uint64_t get_pfn(uint64_t entry)
{
    return ((entry) & 0x3fffffffffffff);
}

//----------------------------------------------------------
/**
 * @brief Converts a virtual address to its physical address
 * 
 * Uses /proc/self/pagemap to translate a virtual address to its corresponding
 * physical address, essential for analyzing DRAM address mappings.
 * 
 * @param v_addr Virtual address to translate
 * @return Corresponding physical address
 */
uint64_t get_phys_addr(uint64_t v_addr)
{
    uint64_t entry;
    uint64_t offset = (v_addr / 4096) * sizeof(entry);  // Calculate pagemap offset
    uint64_t pfn;
    
    // Open pagemap file
    int fd = open("/proc/self/pagemap", O_RDONLY);
    assert(fd >= 0);
    
    // Read the pagemap entry for this virtual address
    int bytes_read = pread(fd, &entry, sizeof(entry), offset);
    close(fd);
    assert(bytes_read == 8);
    assert(entry & (1ULL << 63));  // Ensure page is present
    
    pfn = get_pfn(entry);
    assert(pfn != 0);
    
    // Combine PFN with page offset to get full physical address
    return (pfn * 4096) | (v_addr & 4095);
}

/**
 * @brief Converts a physical address to a virtual address using a PFN-to-VA map
 * 
 * Performs reverse address translation from physical to virtual address
 * using a pre-built mapping table.
 * 
 * @param pa Physical address to convert
 * @param map PFN-to-VA mapping table
 * @param n Number of entries in the mapping table
 * @return Corresponding virtual address, or NULL if not found
 */
void *pa_to_va(uint64_t pa, pfn_va_t *map, size_t n)
{
    uint64_t pfn = pa >> 12;     // Extract PFN from physical address
    uint64_t off = pa & 0xFFF;   // Extract page offset

    /* Linear search is fine for ≤ a few thousand pages */
    for (size_t i = 0; i < n; ++i)
        if (map[i].pfn == pfn)
            return (void *)(map[i].va + off);

    return NULL; /* PFN not found or table stale */
}

//----------------------------------------------------------
/**
 * @brief Creates an address tuple linking virtual and physical addresses
 * 
 * Convenience function to create an addr_tuple structure containing
 * both the virtual address and its corresponding physical address.
 * 
 * @param v_addr Virtual address
 * @return addr_tuple structure with virtual and physical addresses
 */
addr_tuple gen_addr_tuple(char *v_addr)
{
    return (addr_tuple){v_addr, get_phys_addr((uint64_t)v_addr)};
}

/**
 * @brief Performs comprehensive timing measurements for DRAM analysis
 * 
 * This function conducts a systematic timing analysis by:
 * 1. Allocating a large memory buffer with scattered physical addresses
 * 2. Performing thousands of timing measurements between random address pairs
 * 3. Recording timing data along with physical addresses for analysis
 * 4. Saving results to CSV for post-processing and visualization
 * 
 * The timing data reveals patterns in DRAM row buffer behavior, allowing
 * reverse engineering of memory controller address mapping functions.
 * 
 * @param rounds Number of timing measurements per address pair
 * @param m_size Size of memory buffer to allocate in bytes
 * @param o_file Output file path (unused in current implementation)
 * @param flags Memory allocation control flags
 * @param measurements Total number of address pairs to measure
 */
void timing_measurement(size_t rounds, size_t m_size, char *o_file, uint64_t flags, size_t measurements)
{
    // Create a log file named with the memory size for organization
    char log_file_name[50];
    snprintf(log_file_name, sizeof(log_file_name), "data/access_module_%ld.csv", m_size / 1024 / 1024);

    FILE *log_file = fopen(log_file_name, "w");
    if (log_file == NULL)
    {
        fprintf(stderr, "[ERROR] - Unable to create log file\n");
        exit(1);
    }
    // CSV header: physical addresses, timing, virtual addresses
    fprintf(log_file, "a1,a2,elapsed_cycles,v_a1,v_a2\n");

    // Initialize memory buffer structure
    mem_buff_t mem = {
        .buffer = NULL,
        .size = m_size,
        .flags = flags,
    };

    // Allocate the memory buffer with scattered physical pages
    alloc_buffer(&mem);

    // Generate initial random address for baseline measurements
    char *rnd_addr = get_rnd_addr(mem.buffer, mem.size);
    addr_tuple tp = gen_addr_tuple(rnd_addr);

    // Main measurement loop
    for (size_t i = 0; i < measurements; i++)
    {
        char *rnd_addr = get_rnd_addr(mem.buffer, mem.size);
        
        // Perform multiple sub-measurements for each main measurement
        for (int j = 0; j < 50; j++)
        {
            uint64_t time = 0;
            char *rnd_addr2 = get_rnd_addr(mem.buffer, mem.size);
            addr_tuple tmp = gen_addr_tuple(rnd_addr2);
            
            // Take 10 timing samples and record each one
            for (int k = 0; k < 10; k++)
            {
                time = time_tuple((volatile char *)tmp.v_addr, (volatile char *)tp.v_addr, rounds);
                // Log: physical addr1, physical addr2, timing, virtual addr1, virtual addr2
                fprintf(log_file, "%lx,%lx,%ld, %lx, %lx\n", 
                        (uint64_t)tp.p_addr, (uint64_t)tmp.p_addr, time, 
                        (uint64_t)tp.v_addr, (uint64_t)tmp.v_addr);
            }
        }
        
        // Progress reporting every 100 measurements
        if (i % 100 == 0)
        {
            printf("Measurement %ld out of %ld for memory size %ld\n", i, measurements, m_size);
        }
    }
    printf("Results saved to %s\n", log_file_name);
    fclose(log_file);
    free_buffer(&mem);
}





/**
 * @brief Builds a mapping table from physical frame numbers to virtual addresses
 * 
 * Creates a lookup table that maps each physical page frame number (PFN) in
 * the allocated buffer to its corresponding virtual address. This is essential
 * for the bitflip analysis to convert between physical and virtual addresses.
 * 
 * @param buf Base virtual address of the memory buffer
 * @param bytes Size of the memory buffer in bytes
 * @param out_n Output parameter: number of entries in the mapping table
 * @return Pointer to the PFN-to-VA mapping table
 */
pfn_va_t *build_pfn_map(void *buf, size_t bytes, size_t *out_n)
{
    // Open pagemap file for address translation
    int pg = open("/proc/self/pagemap", O_RDONLY);
    assert(pg >= 0);

    size_t pages = bytes >> 12;                 /* 4 KiB pages   */
    pfn_va_t *map = (pfn_va_t *)calloc(pages, sizeof *map); /* dense vector  */

    // Build mapping for each page in the buffer
    for (size_t i = 0; i < pages; ++i)
    {
        uint64_t va = (uint64_t)buf + (i << 12);  // Calculate virtual address
        uint64_t entry;
        
        // Read pagemap entry for this virtual address
        if (pread(pg, &entry, sizeof entry, (va >> 12) * 8) != sizeof(entry))
        {
            perror("Failed to read entry from pagemap");
            exit(EXIT_FAILURE);
        }
        
        // Extract PFN and store in mapping table
        uint64_t pfn = entry & ((1ULL << 55) - 1);
        map[i].pfn = pfn;
        map[i].va = va & ~0xFFFULL;  // Page-aligned virtual address
    }
    close(pg);
    *out_n = pages;
    return map;
}

/**
 * @brief Probes DRAM bank mappings using systematic bit manipulation
 * 
 * This function implements a sophisticated bitflip analysis to reverse engineer
 * the memory controller's bank mapping function. It works by:
 * 
 * 1. Selecting anchor addresses at random locations in physical memory
 * 2. Systematically flipping individual bits and bit combinations in physical addresses
 * 3. Measuring timing differences between anchor and modified addresses
 * 4. Recording patterns that indicate same-bank vs. different-bank relationships
 * 
 * The analysis focuses on bits that are likely to control bank selection,
 * including common bank bits and their combinations. Same-bank addresses
 * will show faster access times due to row buffer locality effects.
 * 
 * @param m_size Size of memory buffer to allocate for analysis
 * @param rounds Number of timing measurements per address pair
 * @param flags Memory allocation control flags
 * @param num_anchors Number of anchor addresses to test from
 */
void mapping_bitflip_probe(size_t m_size, size_t rounds, size_t flags, int num_anchors) {
    printf("=== DRAM Bank XOR Mapping: Same-Row Bank Probing ===\n");

    // Create output file for bitflip analysis results
    char log_file_name[64];
    snprintf(log_file_name, sizeof(log_file_name), "data/bitflip_probe_%lu.csv", m_size / (1024 * 1024));
    FILE *log_file = fopen(log_file_name, "w");
    if (!log_file) {
        fprintf(stderr, "[ERROR] - Could not create log file.\n");
        exit(1);
    }
    // CSV header for bitflip analysis data
    fprintf(log_file, "anchor_va,a1,delta,probe_va,a2,elapsed_cycles\n");

    // Allocate memory buffer for analysis
    mem_buff_t mem = {
        .buffer = NULL,
        .size = m_size,
        .flags = flags,
    };
    alloc_buffer(&mem);

    // Build PFN-to-VA mapping table for address translation
    size_t pfn_map_size;
    pfn_va_t *pfn_map = build_pfn_map(mem.buffer, mem.size, &pfn_map_size);
    printf("[LOG] - PFN map built with %zu entries.\n", pfn_map_size);
    if (pfn_map_size == 0) {
        fprintf(stderr, "[ERROR] - PFN map is empty. Exiting.\n");
        fclose(log_file);
        free_buffer(&mem);
        exit(1);
    }

    // Define candidate bank bits to test (covering typical DRAM address ranges)
    const int bank_bits[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43};
    const int n_bank_bits = sizeof(bank_bits)/sizeof(bank_bits[0]);

    // Test each anchor address
    for (int a_idx = 0; a_idx < num_anchors; a_idx++) {
        // 1️⃣ Pick an anchor address at random
        char *anchor_va = get_rnd_addr(mem.buffer, mem.size);
        uint64_t anchor_pa = get_phys_addr((uint64_t)anchor_va);
        printf("[anchor %d/%d] VA=%p PA=0x%lx\n", a_idx+1, num_anchors, anchor_va, anchor_pa);

        // 2️⃣ Test single-bit flips and combinations with bit 10
        for (int b = 0; b < 33; b++) {
            int i = bank_bits[b];
            
            // Single bit flip test
            uint64_t delta = (1UL << i);
            uint64_t probe_pa = anchor_pa ^ delta;
            char *probe_va = (char *)pa_to_va(probe_pa, pfn_map, pfn_map_size);
            if (probe_va) {
                uint64_t cyc = time_tuple((volatile char*)probe_va, (volatile char*)anchor_va, rounds);
                fprintf(log_file, "%lx,%lx,%lx,%lx,%lx,%lu\n",
                        (uint64_t)anchor_va, anchor_pa,
                        delta,
                        (uint64_t)probe_va, probe_pa,
                        cyc);
            }
            
            // Bit flip combined with bit 10 (often part of bank selection)
            delta = (1UL << i) ^ (1UL << 10);
            probe_pa = anchor_pa ^ delta;
            probe_va = (char *)pa_to_va(probe_pa, pfn_map, pfn_map_size);
            if (probe_va) {
                uint64_t cyc = time_tuple((volatile char*)probe_va, (volatile char*)anchor_va, rounds);
                fprintf(log_file, "%lx,%lx,%lx,%lx,%lx,%lu\n",
                        (uint64_t)anchor_va, anchor_pa,
                        delta,
                        (uint64_t)probe_va, probe_pa,
                        cyc);
            }
        }

        // 3️⃣ Test two-bit flips and their combinations with bit 10
        for (int bi = 0; bi < 33; bi++) {
            for (int bj = 0; bj < bi; bj++) {
                // Two-bit flip pattern
                uint64_t delta = (1UL << bank_bits[bi]) ^ (1UL << bank_bits[bj]);
                
                // Two-bit flip combined with bit 10
                uint64_t delta_with_10 = delta ^ (1UL << 10);

                // Test the two-bit flip
                uint64_t probe_pa = anchor_pa ^ delta;
                char *probe_va = (char *)pa_to_va(probe_pa, pfn_map, pfn_map_size);
                if (probe_va) {
                    uint64_t cyc = time_tuple((volatile char*)probe_va, (volatile char*)anchor_va, rounds);
                    fprintf(log_file, "%lx,%lx,%lx,%lx,%lx,%lu\n",
                            (uint64_t)anchor_va, anchor_pa,
                            delta,
                            (uint64_t)probe_va, probe_pa,
                            cyc);
                }
                
                // Test the two-bit flip combined with bit 10
                probe_pa = anchor_pa ^ delta_with_10;
                probe_va = (char *)pa_to_va(probe_pa, pfn_map, pfn_map_size);
                if (probe_va) {
                    uint64_t cyc = time_tuple((volatile char*)probe_va, (volatile char*)anchor_va, rounds);
                    fprintf(log_file, "%lx,%lx,%lx,%lx,%lx,%lu\n",
                            (uint64_t)anchor_va, anchor_pa,
                            delta_with_10,
                            (uint64_t)probe_va, probe_pa,
                            cyc);
                }
            }
        }
    }

    // Cleanup and completion
    fclose(log_file);
    free_buffer(&mem);
    printf("Bank-probing complete. Data saved to %s\n", log_file_name);
}