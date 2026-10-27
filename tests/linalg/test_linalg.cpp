#include "linalg.h"

#include <cstdio>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static std::vector<bool> bits(uint64_t v, size_t n) {
    std::vector<bool> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = (v >> i) & 1;
    return r;
}

static bool dot_zero(const std::vector<bool>& a, const std::vector<bool>& b) {
    bool acc = false;
    for (size_t i = 0; i < a.size(); ++i) acc ^= (a[i] && b[i]);
    return !acc;
}

// Every nullspace vector must be orthogonal to every input row, and the dimension must be cols - rank.
static void test_nullspace() {
    const size_t cols = 6;
    // rows: x0^x1, x1^x2, x0^x2 (rank 2), x3
    std::vector<std::vector<bool>> rows = {bits(0b000011, cols), bits(0b000110, cols),
                                           bits(0b000101, cols), bits(0b001000, cols)};
    GF2Matrix m(rows.size(), cols, rows);
    auto ns = m.nullspace();
    CHECK(ns.size() == cols - 3);
    for (const auto& v : ns)
        for (const auto& r : rows) CHECK(dot_zero(v, r));
}

static void test_identity_has_empty_nullspace() {
    std::vector<std::vector<bool>> rows = {bits(1, 3), bits(2, 3), bits(4, 3)};
    GF2Matrix m(3, 3, rows);
    CHECK(m.nullspace().empty());
}

int main() {
    test_nullspace();
    test_identity_has_empty_nullspace();
    if (failures) return 1;
    std::puts("linalg tests OK");
    return 0;
}
