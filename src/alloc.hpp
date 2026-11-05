// Small-object allocator for this program (on by default; -DLP_NO_POOL builds with plain malloc).
// Most allocations here are small and short-lived (per-block graphs, paths, table entries). A
// general malloc must handle any pattern; this one keeps, per thread, a free list for each size
// class up to 240 bytes and carves new blocks from 64 KB chunks. Freed blocks go to the freeing
// thread's list; nothing is returned to the OS (the process ends right after the answer).
#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>

namespace lp_alloc {
constexpr std::size_t kHdr = 16, kMaxSmall = 240, kChunk = 64 * 1024;
struct Free {
    Free* next;
};
inline thread_local Free* heads[kMaxSmall / 16 + 1];
inline thread_local char* chunk_ptr = nullptr;
inline thread_local std::size_t chunk_left = 0;

inline void* alloc(std::size_t n) {
    if (n <= kMaxSmall) {
        std::size_t k = (n + 15) / 16;
        if (k == 0) k = 1;
        Free* f = heads[k];
        char* base;
        if (f) {
            heads[k] = f->next;
            base = reinterpret_cast<char*>(f);
        } else {
            std::size_t sz = kHdr + k * 16;
            if (chunk_left < sz) {
                chunk_ptr = static_cast<char*>(std::malloc(kChunk));
                if (!chunk_ptr) throw std::bad_alloc();
                chunk_left = kChunk;
            }
            base = chunk_ptr;
            chunk_ptr += sz, chunk_left -= sz;
        }
        *reinterpret_cast<std::size_t*>(base) = k;
        return base + kHdr;
    }
    char* base = static_cast<char*>(std::malloc(n + kHdr));
    if (!base) throw std::bad_alloc();
    *reinterpret_cast<std::size_t*>(base) = 0;
    return base + kHdr;
}
inline void release(void* p) noexcept {
    if (!p) return;
    char* base = static_cast<char*>(p) - kHdr;
    std::size_t k = *reinterpret_cast<std::size_t*>(base);
    if (k == 0) {
        std::free(base);
        return;
    }
    Free* f = reinterpret_cast<Free*>(base);
    f->next = heads[k];
    heads[k] = f;
}
}  // namespace lp_alloc

void* operator new(std::size_t n) { return lp_alloc::alloc(n); }
void* operator new[](std::size_t n) { return lp_alloc::alloc(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return lp_alloc::alloc(n);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return lp_alloc::alloc(n);
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* p) noexcept { lp_alloc::release(p); }
void operator delete[](void* p) noexcept { lp_alloc::release(p); }
void operator delete(void* p, std::size_t) noexcept { lp_alloc::release(p); }
void operator delete[](void* p, std::size_t) noexcept { lp_alloc::release(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { lp_alloc::release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { lp_alloc::release(p); }
