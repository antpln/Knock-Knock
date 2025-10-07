#include "linalg.h"
#include <algorithm>
#include <iostream>

// Convert bool vector to uint64_t row representation
static inline uint64_t bool_vec_to_uint64(const std::vector<bool>& vec) {
    uint64_t result = 0;
    const size_t limit = std::min<size_t>(64, vec.size());
    for (size_t i = 0; i < limit; ++i) {
        if (vec[i]) {
            result |= (1ULL << i);
        }
    }
    return result;
}

// Convert uint64_t row to bool vector
static inline std::vector<bool> uint64_to_bool_vec(uint64_t row, size_t cols) {
    std::vector<bool> result(cols, false);
    const size_t limit = std::min<size_t>(64, cols);
    for (size_t i = 0; i < limit; ++i) {
        if (row & (1ULL << i)) {
            result[i] = true;
        }
    }
    return result;
}

// Get bit at position
static inline bool get_bit(uint64_t row, size_t pos) {
    return (row & (1ULL << pos)) != 0;
}

GF2Matrix::GF2Matrix(size_t rows, size_t cols, const std::vector<std::vector<bool>>& data)
    : rows_(rows), cols_(cols), matrix_rows_(rows) {
    // Convert bool vectors to uint64_t representation
    for (size_t i = 0; i < rows; ++i) {
        matrix_rows_[i] = bool_vec_to_uint64(data[i]);
    }
}

std::vector<bool> GF2Matrix::get_row(size_t i) const {
    return uint64_to_bool_vec(matrix_rows_[i], cols_);
}

void GF2Matrix::row_reduce() {
    size_t pivot_row = 0;
    
    for (size_t j = 0; j < cols_ && pivot_row < rows_; ++j) {
        // Find pivot: first row with bit j set, starting from pivot_row
        size_t i = pivot_row;
        while (i < rows_ && !get_bit(matrix_rows_[i], j)) {
            i++;
        }

        if (i < rows_) {
            // Swap rows to bring pivot to pivot_row position
            std::swap(matrix_rows_[pivot_row], matrix_rows_[i]);
            
            // Eliminate bit j in all other rows using XOR
            // This is where uint64_t shines: single XOR operation for entire row
            const uint64_t pivot_row_data = matrix_rows_[pivot_row];
            
            for (i = 0; i < rows_; ++i) {
                if (i != pivot_row && get_bit(matrix_rows_[i], j)) {
                    // XOR entire row in one operation!
                    matrix_rows_[i] ^= pivot_row_data;
                }
            }
            pivot_row++;
        }
    }
}

std::vector<std::vector<bool>> GF2Matrix::nullspace() {
    row_reduce();

    // Find pivot columns (columns with leading 1 in some row)
    std::vector<int> pivot_cols(rows_, -1);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (get_bit(matrix_rows_[i], j)) {
                pivot_cols[i] = j;
                break;
            }
        }
    }

    // Find free columns (not pivot columns)
    std::vector<int> free_cols;
    for (size_t j = 0; j < cols_; ++j) {
        bool is_pivot = false;
        for (size_t i = 0; i < rows_; ++i) {
            if (pivot_cols[i] == static_cast<int>(j)) {
                is_pivot = true;
                break;
            }
        }
        if (!is_pivot) {
            free_cols.push_back(j);
        }
    }

    // Build nullspace basis: one vector for each free column
    std::vector<std::vector<bool>> basis;
    basis.reserve(free_cols.size());
    
    for (int free_col : free_cols) {
        // Start with free variable set to 1
        uint64_t vec_as_uint64 = (1ULL << free_col);
        
        // For each row with a pivot, set the pivot column bit if needed
        for (size_t i = 0; i < rows_; ++i) {
            if (pivot_cols[i] != -1 && get_bit(matrix_rows_[i], free_col)) {
                // Set the pivot column bit to maintain nullspace property
                vec_as_uint64 |= (1ULL << pivot_cols[i]);
            }
        }
        
        basis.push_back(uint64_to_bool_vec(vec_as_uint64, cols_));
    }

    return basis;
}
