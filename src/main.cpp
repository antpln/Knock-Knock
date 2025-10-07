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
    fprintf(stderr, "DRAM Reverse Engineering Tool - Knock-Knock Attack Implementation\n\n");
    fprintf(stderr, "Usage: ./main [OPTIONS]\n\n");
    fprintf(stderr, "Basic Options:\n");
    fprintf(stderr, "  -h, --help                     Show this help message\n");
    fprintf(stderr, "  -v                             Verbose output\n");
    fprintf(stderr, "  --full-analysis                Run full automated analysis (default mode)\n\n");
    
    fprintf(stderr, "Memory Configuration:\n");
    fprintf(stderr, "  -m <MB>                        Memory size in MB (default: %d MB)\n", MEM_SIZE_MB_std);
    fprintf(stderr, "  -p <percent>                   Memory size as %% of total (overrides -m)\n");
    fprintf(stderr, "  -d, --delay <us>               Delay in microseconds after each measurement\n");
    fprintf(stderr, "                                 (helps avoid tFAW violations, default: 0)\n\n");
    
    fprintf(stderr, "Threshold Detection Options:\n");
    fprintf(stderr, "  --threshold <cycles>           Manually set conflict threshold (skip auto-detection)\n");
    fprintf(stderr, "  --threshold-samples <n>        Number of samples for threshold detection (default: 100000)\n\n");
    
    fprintf(stderr, "Bank Mask Detection Options:\n");
    fprintf(stderr, "  --bank-conflicts <n>           Target number of conflict samples (default: 5000)\n");
    fprintf(stderr, "  --bank-measurements <n>        Max measurements for bank detection (default: 3000000)\n");
    fprintf(stderr, "  --bank-subsample <n>           Subsample size for nullspace analysis (default: 1000)\n");
    fprintf(stderr, "  --bank-rounds <n>              Number of subsampling rounds (default: 35)\n");
    fprintf(stderr, "  --bank-attempts <n>            Max retry attempts if first round fails (default: 3)\n\n");
    
    fprintf(stderr, "Row Mask Detection Options:\n");
    fprintf(stderr, "  --row-pairs <n>                Target number of same-bank pairs (default: 8000)\n");
    fprintf(stderr, "  --row-min-hits <n>             Minimum hit samples required (default: 1000)\n");
    fprintf(stderr, "  --row-min-conflicts <n>        Minimum conflict samples required (default: 1000)\n");
    fprintf(stderr, "  --row-max-attempts <n>         Max attempts for row sampling (default: 2400000)\n\n");
    
    fprintf(stderr, "Advanced Options:\n");
    fprintf(stderr, "  --force-multiple-rounds        Force multiple detection rounds even if first succeeds\n\n");
    
    fprintf(stderr, "Output Options (legacy):\n");
    fprintf(stderr, "  -o <file>                      Output file for measurements (default: %s)\n", O_FILE_std);
    fprintf(stderr, "  -a                             Auto-name output as <hostname>_<size_mb>.csv\n");
    fprintf(stderr, "  --timing                       Run timing measurement mode\n");
    fprintf(stderr, "  --bitflip                      Run mapping bitflip probe\n\n");
    
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  # Run with default settings:\n");
    fprintf(stderr, "  sudo ./main --full-analysis\n\n");
    fprintf(stderr, "  # Use 50%% of system memory:\n");
    fprintf(stderr, "  sudo ./main --full-analysis -p 50\n\n");
    fprintf(stderr, "  # Manual threshold and custom bank detection:\n");
    fprintf(stderr, "  sudo ./main --full-analysis --threshold 180 --bank-conflicts 10000 --bank-rounds 60\n\n");
    fprintf(stderr, "  # High-precision row detection:\n");
    fprintf(stderr, "  sudo ./main --full-analysis --row-pairs 15000 --row-min-hits 2000\n\n");
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

    // Analysis configuration with defaults
    AnalysisConfig analysis_config;

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
              {"help", no_argument, 0, 'h'},
              // Threshold options
              {"threshold", required_argument, 0, 1000},
              {"threshold-samples", required_argument, 0, 1001},
              // Bank mask options
              {"bank-conflicts", required_argument, 0, 1010},
              {"bank-measurements", required_argument, 0, 1011},
              {"bank-subsample", required_argument, 0, 1012},
              {"bank-rounds", required_argument, 0, 1013},
              {"bank-attempts", required_argument, 0, 1014},
              // Row mask options
              {"row-pairs", required_argument, 0, 1020},
              {"row-min-hits", required_argument, 0, 1021},
              {"row-min-conflicts", required_argument, 0, 1022},
              {"row-max-attempts", required_argument, 0, 1023},
              // Advanced options
              {"force-multiple-rounds", no_argument, 0, 1030},
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
            
            // Threshold detection options
            case 1000: // --threshold
                analysis_config.manual_threshold = (uint64_t) atol(optarg);
                if (analysis_config.manual_threshold <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid threshold: %s\n", optarg);
                    return 1;
                }
                break;
            case 1001: // --threshold-samples
                analysis_config.threshold_samples = (size_t) atol(optarg);
                if (analysis_config.threshold_samples <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid threshold samples: %s\n", optarg);
                    return 1;
                }
                break;
            
            // Bank mask options
            case 1010: // --bank-conflicts
                analysis_config.bank_target_conflicts = (size_t) atol(optarg);
                if (analysis_config.bank_target_conflicts <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid bank conflicts: %s\n", optarg);
                    return 1;
                }
                break;
            case 1011: // --bank-measurements
                analysis_config.bank_max_measurements = (size_t) atol(optarg);
                if (analysis_config.bank_max_measurements <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid bank measurements: %s\n", optarg);
                    return 1;
                }
                break;
            case 1012: // --bank-subsample
                analysis_config.bank_subsample_size = (size_t) atol(optarg);
                if (analysis_config.bank_subsample_size <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid bank subsample size: %s\n", optarg);
                    return 1;
                }
                break;
            case 1013: // --bank-rounds
                analysis_config.bank_repeat_rounds = (size_t) atol(optarg);
                if (analysis_config.bank_repeat_rounds <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid bank rounds: %s\n", optarg);
                    return 1;
                }
                break;
            case 1014: // --bank-attempts
                analysis_config.bank_max_attempts = (size_t) atol(optarg);
                if (analysis_config.bank_max_attempts <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid bank attempts: %s\n", optarg);
                    return 1;
                }
                break;
            
            // Row mask options
            case 1020: // --row-pairs
                analysis_config.row_target_pairs = (size_t) atol(optarg);
                if (analysis_config.row_target_pairs <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid row pairs: %s\n", optarg);
                    return 1;
                }
                break;
            case 1021: // --row-min-hits
                analysis_config.row_min_hits = (size_t) atol(optarg);
                if (analysis_config.row_min_hits <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid row min hits: %s\n", optarg);
                    return 1;
                }
                break;
            case 1022: // --row-min-conflicts
                analysis_config.row_min_conflicts = (size_t) atol(optarg);
                if (analysis_config.row_min_conflicts <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid row min conflicts: %s\n", optarg);
                    return 1;
                }
                break;
            case 1023: // --row-max-attempts
                analysis_config.row_max_attempts = (size_t) atol(optarg);
                if (analysis_config.row_max_attempts <= 0) {
                    fprintf(stderr, "[ERROR] - Invalid row max attempts: %s\n", optarg);
                    return 1;
                }
                break;
            
            // Advanced options
            case 1030: // --force-multiple-rounds
                analysis_config.force_multiple_rounds = true;
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
        FullAnalysis analysis(m_size, delay_us, analysis_config);
        analysis.run();
    } else {
        // Default to full analysis
        FullAnalysis analysis(m_size, delay_us, analysis_config);
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
