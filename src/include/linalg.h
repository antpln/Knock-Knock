#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cstddef>

class GF2Matrix {
public:
    GF2Matrix(size_t rows, size_t cols, const std::vector<std::vector<bool>>& data);
    
    void row_reduce();
    std::vector<std::vector<bool>> nullspace();

private:
    size_t rows_;
    size_t cols_;
    std::vector<std::vector<bool>> matrix_;
};

#endif // LINALG_H
