#pragma once
// MTL5 -- Vector element inserter for sparse vector construction
// Port from MTL4: boost/numeric/mtl/vector/inserter.hpp
// Reuses update_store/update_plus from mat/inserter.hpp

#include <cstddef>
#include <mtl/mat/inserter.hpp>  // for update_store, update_plus
#include <mtl/vec/sparse_vector.hpp>

namespace mtl::vec {

/// Inserter for sparse_vector -- provides proxy-based element insertion.
/// Unlike the matrix inserter, vector insertion is immediate (no deferred finalize).
template <typename Value, typename Parameters = parameters<>,
          typename Updater = mat::update_store<Value>>
class sparse_vector_inserter {
    using vector_type = sparse_vector<Value, Parameters>;
    using size_type   = typename vector_type::size_type;

    /// Proxy returned by operator[] -- supports << for insertion
    struct update_proxy {
        vector_type& vec_;
        size_type    idx_;
        Updater      updater_;

        update_proxy& operator<<(const Value& val) {
            updater_(vec_[idx_], val);
            return *this;
        }

        update_proxy& operator=(const Value& val) {
            vec_[idx_] = val;
            return *this;
        }

        update_proxy& operator+=(const Value& val) {
            vec_[idx_] += val;
            return *this;
        }
    };

public:
    explicit sparse_vector_inserter(vector_type& vec)
        : vec_(vec), updater_{} {}

    // Non-copyable
    sparse_vector_inserter(const sparse_vector_inserter&) = delete;
    sparse_vector_inserter& operator=(const sparse_vector_inserter&) = delete;

    update_proxy operator[](size_type i) {
        return update_proxy{vec_, i, updater_};
    }

private:
    vector_type& vec_;
    Updater updater_;
};

/// Convenience alias: vec_inserter<VecType, Updater>
template <typename Vec, typename Updater = mat::update_store<typename Vec::value_type>>
using vec_inserter = sparse_vector_inserter<
    typename Vec::value_type,
    typename Vec::param_type,
    Updater>;

} // namespace mtl::vec
