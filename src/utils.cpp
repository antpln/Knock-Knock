/**
 * @file utils.cpp
 * @brief Utility functions for memory allocation and management
 * 
 * Implements memory allocation functions optimized for DRAM reverse engineering,
 * including support for scattered physical page allocation and huge pages.
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <linux/mman.h>
#include <time.h>

#include "utils.h"


//-----------------------------------------------
// 			Memory allocation functions

/**
 * @brief Fisher-Yates shuffle algorithm for randomizing page order
 * 
 * Randomly shuffles an array of page indices to ensure pages are faulted
 * in random order, promoting scattered physical page allocation.
 * 
 * @param idx Array of page indices to shuffle
 * @param n Number of elements in the array
 */
static void shuffle_pages(size_t *idx, size_t n)
{
    srand((unsigned)time(NULL));
    for (size_t i = n - 1; i; --i) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

/**
 * @brief Allocate a memory buffer optimized for DRAM analysis
 * 
 * Allocates memory with specific properties beneficial for DRAM reverse engineering:
 * - Uses anonymous memory mapping for control over physical allocation
 * - Disables transparent huge pages for consistent 4KB granularity
 * - Faults pages in random order to promote scattered physical addresses
 * - Supports various allocation flags for different analysis needs
 * 
 * @param mem Pointer to mem_buff_t structure with allocation parameters
 * @return 0 on success, -1 on failure
 */
int alloc_buffer(mem_buff_t *mem)
{
    // Validate input parameters
    if (mem->buffer) {
        fprintf(stderr, "[ERROR] buffer already allocated\n");
        return -1;
    }
    if (mem->size == 0 || (mem->size % KB(4))) {
        fprintf(stderr, "[ERROR] size must be >0 and 4-KiB aligned\n");
        return -1;
    }

    /* 1. Reserve anonymous virtual space (lazy allocation). */
    mem->buffer = (char*)mmap(NULL, mem->size,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | mem->flags,
                       -1, 0);
    if (mem->buffer == MAP_FAILED) {
        perror("[ERROR] mmap");
        return -1;
    }

    /* 2. Never promote to THP, we want 4-KiB granularity for precise analysis. */
    if (madvise(mem->buffer, mem->size, MADV_NOHUGEPAGE) != 0)
        perror("[WARN] madvise(MADV_NOHUGEPAGE)");

    /* 3. Fault pages in *random order* so buddy allocator
          pulls from many free lists -> scattered PFNs for better analysis. */
    const size_t npages = mem->size / KB(4);
    size_t *order = (size_t *)malloc(npages * sizeof(size_t));
    if (!order) { perror("malloc"); return -1; }

    // Initialize page indices
    for (size_t i = 0; i < npages; ++i) order[i] = i;
    // Randomize the order
    shuffle_pages(order, npages);

    // Fault pages in random order to scatter physical addresses
    for (size_t k = 0; k < npages; ++k) {
        volatile char *p = mem->buffer + order[k] * KB(4);
        *p = 0;                        /* one write => one page-fault */
    }
    free(order);

    fprintf(stderr,
            "[LOG] allocated %.2f MiB (%zu pages) at %p "
            "(4-KiB pages, NUMA-interleaved, shuffled)\n",
            mem->size / (double)MB(1), npages, mem->buffer);
    return 0;
}

/**
 * @brief Free a previously allocated memory buffer
 * 
 * Unmaps the memory region allocated by alloc_buffer().
 * 
 * @param mem Pointer to mem_buff_t structure containing buffer to free
 * @return 0 on success, -1 on failure
 */
int free_buffer(mem_buff_t* mem) {
	return munmap(mem->buffer, mem->size);
}

/**
 * @brief Convert a virtual address to a physical address
 * 
 * This function uses the /proc/self/pagemap interface to translate a virtual
 * address to its corresponding physical address. It is primarily used for
 * debugging and analysis purposes, to verify the physical memory mapping
 * of allocated buffers.
 * 
 * @param v_addr The virtual address to convert
 * @return The corresponding physical address, or 0 on failure
 */
uint64_t virt_to_phys(uint64_t v_addr) {
    long page_size = sysconf(_SC_PAGESIZE);
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        return 0;
    }
    uint64_t p_addr = 0;
    uint64_t offset = (v_addr / page_size) * sizeof(uint64_t);
    if (lseek(fd, offset, SEEK_SET) < 0) {
        close(fd);
        return 0;
    }
    uint64_t entry;
    if (read(fd, &entry, sizeof(uint64_t)) != sizeof(uint64_t)) {
        close(fd);
        return 0;
    }
    close(fd);

    if (!(entry & (1ULL << 63))) {
        return 0;
    }

    p_addr = (entry & ((1ULL << 55) - 1)) * page_size + (v_addr % page_size);
    return p_addr;
}
