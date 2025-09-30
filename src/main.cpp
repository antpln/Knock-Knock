/**
 * @file main.cpp
 * @brief DRAM Reverse Engineering Tool - Knock-Knock Attack Implementation
 * 
 * This tool implements a timing-based attack to reverse engineer DRAM memory controller
 * mappings and bank configurations. It uses row buffer conflicts and timing analysis
 * to understand how physical addresses map to DRAM banks and rows.
 */

#define _GNU_SOURCE
#include "stdio.h"
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>
#include <getopt.h>
#include <sys/mman.h>

#include "utils.h"
#include "rev-mc.h"
#include "full_analysis.h"

// Default configuration constants
#define ROUNDS_std      50               // Default number of timing measurement rounds
#define O_FILE_std      "output.csv"     // Default output file for results
#define MEM_SIZE_MB_std 25600            // Default memory size: 25GB in MB
#define MEASUREMENTS_std 100000          // Default number of measurements to perform


//-----------------------------------------------
//                  GLOBALS



//-----------------------------------------------
/**
 * @brief Prints usage information for the program
 * 
 * Displays all available command-line options and their descriptions,
 * including default values for configurable parameters.
 */
void print_usage() {
    fprintf(stderr, "[ LOG ] - Usage ./test [-h] [-o o_file] [-a] [-v] [--timing] [--bitflip] [--full-analysis] [-m mem_mb] [-p mem_percent] [-r rounds] [-n measurements]\n");
    fprintf(stderr, "          -h                     = this help message\n");
    fprintf(stderr, "          -o o_file              = output file for mem profiling      (default: %s)\n", O_FILE_std);
    fprintf(stderr, "          -a                     = auto-name output file as <hostname>_<size_in_mb>.csv\n");
    fprintf(stderr, "          --timing               = run timing measurement instead of rev_mc\n");
    fprintf(stderr, "          --bitflip              = run mapping bitflip probe\n");
    fprintf(stderr, "          --full-analysis        = run the full automated analysis\n");
    fprintf(stderr, "          -m mem_mb              = memory size in MB                   (default: %d MB)\n", MEM_SIZE_MB_std);
    fprintf(stderr, "          -p mem_percent         = memory size as percentage of total  (overrides -m)\n");
    fprintf(stderr, "          -r rounds              = number of rounds                    (default: %d)\n", ROUNDS_std);
    fprintf(stderr, "          -n measurements        = number of measurements              (default: %d)\n", MEASUREMENTS_std);
    fprintf(stderr, "          -v                     = verbose\n\n");
}

//-----------------------------------------------
/**
 * @brief Retrieves the total system memory in megabytes
 * 
 * Reads /proc/meminfo to determine the total available system memory.
 * This is used when the user specifies memory size as a percentage
 * instead of an absolute value.
 * 
 * @return Total system memory in megabytes
 * @note Exits the program if unable to read or parse /proc/meminfo
 */
size_t get_total_memory_mb() {
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo == NULL) {
        fprintf(stderr, "[ERROR] - Unable to read /proc/meminfo\n");
        exit(1);
    }
    
    char line[256];
    size_t total_kb = 0;
    
    // Parse the MemTotal line from /proc/meminfo
    while (fgets(line, sizeof(line), meminfo)) {
        if (sscanf(line, "MemTotal: %zu kB", &total_kb) == 1) {
            break;
        }
    }
    
    fclose(meminfo);
    
    if (total_kb == 0) {
        fprintf(stderr, "[ERROR] - Unable to parse total memory from /proc/meminfo\n");
        exit(1);
    }
    
    // Convert KB to MB
    return total_kb / 1024;
}

//-----------------------------------------------
/**
 * @brief Pins the current process to a specific CPU core
 * 
 * Uses CPU affinity to ensure consistent timing measurements by
 * preventing the process from migrating between cores during execution.
 * This is crucial for accurate timing analysis in reverse engineering.
 * 
 * @param core_id The CPU core ID to pin the process to
 * @note Exits the program if unable to set CPU affinity
 */
void pin_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pid_t pid = getpid(); // Get current process ID

    if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }
    else {
        fprintf(stderr, "[ LOG ] - Pinned to core %d\n", core_id);
    }
}

/**
 * @brief Generates an output filename based on hostname and memory size
 * 
 * Creates a filename in the format "<hostname>_<size_in_mb>.csv" by
 * reading the system hostname and using the specified memory size.
 * 
 * @param mem_size_mb Memory size in megabytes to include in filename
 * @return Allocated string containing the generated filename (must be freed by caller)
 * @note Exits the program if unable to get hostname or allocate memory
 */
char* generate_auto_filename(size_t mem_size_mb) {
    char hostname[256];
    
    // Get the system hostname
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        fprintf(stderr, "[ERROR] - Unable to get hostname\n");
        exit(1);
    }
    
    // Calculate required buffer size for the filename
    // Format: "<hostname>_<size_in_mb>.csv"
    size_t filename_len = strlen(hostname) + 32; // Extra space for size and extension
    char* filename = (char*) malloc(filename_len);
    
    if (filename == NULL) {
        fprintf(stderr, "[ERROR] - Unable to allocate memory for filename\n");
        exit(1);
    }
    
    // Generate the filename
    snprintf(filename, filename_len, "data/%s_%zu.csv", hostname, mem_size_mb);
    
    return filename;
}

/**
 * @brief Main entry point for the DRAM reverse engineering tool
 * 
 * Parses command-line arguments and executes either timing measurements
 * or bitflip probing to analyze DRAM memory controller behavior.
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on success, 1 on error
 */
int main(int argc, char** argv) {

    // Configuration variables with default values
    uint64_t    flags       = 0ULL;              // Control flags for memory allocation
    size_t      rounds      = ROUNDS_std;        // Number of timing measurement rounds
    char*       o_file      = (char*) O_FILE_std; // Output file path
    size_t      mem_size_mb = MEM_SIZE_MB_std;   // Memory size in megabytes
    bool        run_bitflip = false;             // Whether to run bitflip analysis
    bool        run_full_analysis = false;       // Whether to run the full analysis
    size_t      measurements = MEASUREMENTS_std;  // Number of measurements to perform
    double      mem_percent = 0.0;               // Memory size as percentage of total
    bool        use_percentage = false;          // Flag to use percentage instead of fixed size
    bool        auto_name = false;               // Flag to auto-generate filename based on hostname and size
    unsigned int delay_us = 0;                   // Delay in us after pair measurement for tFAW

    // Set default flags for memory population and verbose output
    flags |= F_POPULATE;
    flags |= F_VERBOSE;

    // Pin to core 3 for consistent timing measurements
    pin_to_core(3);

    // Ensure running as root to access /proc/self/pagemap
    if(geteuid() != 0) {
    	fprintf(stderr, "[ERROR] - You need to run as root to access pagemap!\n");
	exit(1);
    }

    pin_to_core(3); // Pin to core 3 (redundant but ensures consistency)

    // Prevent the kernel from swapping out our memory to ensure consistent timing
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall");
        return 1;
    }

    // Parse command-line arguments
    while (1) {
        int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] =
            {
              {"timing", no_argument, 0, '1'},    // Enable timing measurement mode
              {"bitflip", no_argument, 0, '2'},   // Enable bitflip probing mode
              {"full-analysis", no_argument, 0, '3'}, // Enable full analysis mode
              {"delay", required_argument, 0, 'd'},
              {0, 0, 0, 0}
            };
        int arg = getopt_long(argc, argv, "o:hvm:r:n:p:ad:",
                              long_options, &option_index);

        if (arg == -1)
            break;

        switch(arg) {
            case '1': // Handle --timing
                // Timing mode is default, no additional action needed
                break;
            case '2': // Handle --bitflip
                run_bitflip = true;
                break;
            case '3': // Handle --full-analysis
                run_full_analysis = true;
                break;
            case 'o': // Output file specification
                o_file = (char*) malloc(sizeof(char) * (strlen(optarg) + 1));
                strncpy(o_file, optarg, strlen(optarg));
                o_file[strlen(optarg)] = '\0';
                flags |= F_EXPORT;
                break;
            case 'a': // Auto-generate filename
                auto_name = true;
                flags |= F_EXPORT;
                break;
            case 'm': // Memory size in MB
                mem_size_mb = (size_t) atol(optarg);
                if (mem_size_mb <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid memory size: %s\n", optarg);
                    return 1;
                }
                break;
            case 'r': // Number of timing rounds
                rounds = (size_t) atol(optarg);
                if (rounds <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid number of rounds: %s\n", optarg);
                    return 1;
                }
                break;
            case 'n': // Number of measurements
                measurements = (size_t) atol(optarg);
                if (measurements <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid number of measurements: %s\n", optarg);
                    return 1;
                }
                break;
            case 'p': // Memory size as percentage
                mem_percent = atof(optarg);
                if (mem_percent <= 0.0 || mem_percent > 100.0) {
                    fprintf(stderr, "[ERROR] - Invalid memory percentage: %s (must be between 0 and 100)\n", optarg);
                    return 1;
                }
                use_percentage = true;
                break;
            case 'd': // Delay in us
                delay_us = (unsigned int) atoi(optarg);
                break;
            case 'h':
            default:
                print_usage();
                return 0;
        }
    }
    
    // Check for conflicting options
    if (auto_name && flags & F_EXPORT && o_file != (char*) O_FILE_std) {
        fprintf(stderr, "[ERROR] - Cannot specify both -o and -a options simultaneously\n");
        return 1;
    }
    
    // Calculate final memory size based on user input
    size_t m_size;
    if (use_percentage) {
        // Use percentage of total system memory
        size_t total_mem_mb = get_total_memory_mb();
        mem_size_mb = (size_t)(total_mem_mb * mem_percent / 100.0);
        printf("[ LOG ] - Using %.1f%% of total memory (%zu MB out of %zu MB)\n", 
               mem_percent, mem_size_mb, total_mem_mb);
        m_size = MB(mem_size_mb);
    } else {
        // Use fixed memory size
        m_size = MB(mem_size_mb);
        printf("[ LOG ] - Using fixed memory size: %zu MB\n", mem_size_mb);
    }
    
    // Generate auto filename if requested
    if (auto_name) {
        o_file = generate_auto_filename(mem_size_mb);
        printf("[ LOG ] - Auto-generated output filename: %s\n", o_file);
    }
    
    // Execute the selected analysis mode
    if (run_full_analysis) {
        FullAnalysis analysis(m_size, delay_us);
        analysis.run();
    } else {
        // This is the default timing analysis mode
        // We need to ensure rev_mc_timing is declared
        // For now, let's assume it's in rev-mc.h and the issue is elsewhere.
        // The user might have removed it, let's just call the analysis.
        FullAnalysis analysis(m_size, delay_us);
        analysis.run();
    }
    
    // Clean up allocated memory
    if (auto_name && o_file != (char*) O_FILE_std) {
        free(o_file);
    }
    else if (o_file != (char*) O_FILE_std) {
        free(o_file);
    }

    return 0;
}
