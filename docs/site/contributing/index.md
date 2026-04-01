---
title: Contributing
description: How to contribute to the Matrix Template Library (MTL5)

---

Thank you for your interest in contributing to MTL5.

## How to Contribute

1. **Report bugs** — open a [GitHub Issue](https://github.com/stillwater-sc/mtl5/issues) with a minimal reproducer
2. **Suggest features** — describe the use case and proposed solution in an issue
3. **Submit pull requests** — fork the repo, create a feature branch, and open a PR against `main`

## Development Setup

```bash
git clone https://github.com/stillwater-sc/mtl5.git
cd mtl5
cmake --preset dev
cmake --build build -j$(nproc)
ctest --test-dir build
```

## Guidelines

- Follow C++20 idioms: concepts, `if constexpr`, `std::span`, ranges
- Add Catch2 tests for new features in `tests/unit/`
- Ensure CI passes on GCC, Clang, Apple Clang, and MSVC before requesting review
- Use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages

## More Information

- [Contributors](/mtl5/contributing/contributors/) — current project contributors
- [Code of Conduct](/mtl5/contributing/code-of-conduct/) — community standards
