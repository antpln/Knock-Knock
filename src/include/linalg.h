#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cstddef>
#include <cstdint>

class GF2Matrix {
public:
    GF2Matrix(size_t rows, size_t cols, const std::vector<std::vector<bool>>& data);
    
    void row_reduce();
    std::vector<std::vector<bool>> nullspace();
    
    // For testing
    std::vector<bool> get_row(size_t i) const;

private:
    size_t rows_;
    size_t cols_;
    
    // Optimized representation: each row is a uint64_t bitmask
    // This allows 64 simultaneous XOR operations
    std::vector<uint64_t> matrix_rows_;
};

#endif // LINALG_H
