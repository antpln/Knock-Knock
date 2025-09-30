#include "linalg.h"
#include <algorithm>
#include <iostream>

GF2Matrix::GF2Matrix(size_t rows, size_t cols, const std::vector<std::vector<bool>>& data)
    : rows_(rows), cols_(cols), matrix_(data) {}

void GF2Matrix::row_reduce() {
    size_t pivot_row = 0;
    for (size_t j = 0; j < cols_ && pivot_row < rows_; ++j) {
        size_t i = pivot_row;
        while (i < rows_ && !matrix_[i][j]) {
            i++;
        }

        if (i < rows_) {
            std::swap(matrix_[pivot_row], matrix_[i]);
            for (i = 0; i < rows_; ++i) {
                if (i != pivot_row && matrix_[i][j]) {
                    for (size_t k = 0; k < cols_; ++k) {
                        matrix_[i][k] = matrix_[i][k] ^ matrix_[pivot_row][k];
                    }
                }
            }
            pivot_row++;
        }
    }
}

std::vector<std::vector<bool>> GF2Matrix::nullspace() {
    row_reduce();

    std::vector<int> pivot_cols(rows_, -1);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (matrix_[i][j]) {
                pivot_cols[i] = j;
                break;
            }
        }
    }

    std::vector<int> free_cols;
    for (size_t j = 0; j < cols_; ++j) {
        bool is_pivot = false;
        for (size_t i = 0; i < rows_; ++i) {
            if (pivot_cols[i] == (int)j) {
                is_pivot = true;
                break;
            }
        }
        if (!is_pivot) {
            free_cols.push_back(j);
        }
    }

    std::vector<std::vector<bool>> basis;
    for (int free_col : free_cols) {
        std::vector<bool> vec(cols_, false);
        vec[free_col] = true;
        for (size_t i = 0; i < rows_; ++i) {
            if (pivot_cols[i] != -1 && matrix_[i][free_col]) {
                vec[pivot_cols[i]] = true;
            }
        }
        basis.push_back(vec);
    }

    return basis;
}
