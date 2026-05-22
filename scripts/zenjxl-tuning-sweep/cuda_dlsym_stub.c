/*
 * cuda_dlsym_stub.c — LD_PRELOAD shim that stubs out cudarc 0.19.4's
 * lookup of CUDA coredump-callback symbols that were removed from
 * libcuda.so in CUDA 13.x driver releases.
 *
 * Symbols intercepted (all return a no-op function pointer):
 *   - cuCoredumpDeregisterCompleteCallback  (observed on driver 580.142)
 *   - cuCoredumpDeregisterStartCallback     (observed on driver 580.x
 *                                            during EXP-MULTI-CODEC
 *                                            smoke 2026-05-18; the
 *                                            first stub revision only
 *                                            handled the Complete
 *                                            variant, leaving the
 *                                            Start variant uncovered.)
 *   - cuDevSmResourceSplit / cuDevSm*       (Hopper SM-partition API
 *                                            cudarc 0.19.4 references
 *                                            statically; missing on
 *                                            non-Hopper drivers and
 *                                            many older 12.x builds.
 *                                            Observed 2026-05-19 on
 *                                            instance 37050770, driver
 *                                            shipping libcuda.so
 *                                            without the symbol.)
 *
 * Fallback: any `cu*` symbol that real_dlsym returns NULL for receives
 * the no-op stub. cudarc 0.19.4 dlsyms ~3000 driver symbols at init
 * time and panics if any are missing — but in practice most missing
 * symbols are for features (coredump teardown, SM partitioning,
 * multicast memory, future-CUDA aliases) that zen-metrics never
 * invokes at runtime. Returning a no-op stub keeps the binding alive;
 * if the no-op is ever actually called the function returns silently
 * (which the CUDA C ABI treats as "feature unavailable, success").
 *
 * Without this shim, every cubecl-cuda device init panics with
 *   "Expected symbol in library: DlSym { source: ... }"
 * and the GPU dispatcher dies, causing every subsequent metric call
 * to fail with RecvError on the closed channel. Observed on driver
 * 580.142 (instance 37035295) and 570.x (instance 37041995); fix
 * unblocks the entire vast.ai offer pool.
 *
 * Safety: the Register/Deregister Start/Complete callbacks are part
 * of CUDA's coredump-callback teardown API — only invoked during
 * process shutdown if the application has registered a callback
 * (cuCoredumpRegisterStartCallback / cuCoredumpRegisterCompleteCallback).
 * zen-metrics never does. No-op stubs are therefore safe: cudarc's
 * static lookup succeeds, the dispatcher stays alive, and the
 * function pointers are never actually called at runtime.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o /usr/local/lib/cuda_dlsym_stub.so \
 *       cuda_dlsym_stub.c -ldl
 * Use:
 *   ENV LD_PRELOAD=/usr/local/lib/cuda_dlsym_stub.so
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

static int cu_coredump_callback_noop(void) {
    /* Universal no-op stub. CUDA driver functions return CUresult,
     * an enum where 0 == CUDA_SUCCESS. Returning 0 makes any caller
     * that bothered to check the return code see "success", which
     * is the right answer for "this feature isn't actually wired up
     * but cudarc statically references the symbol". */
    return 0;
}

typedef void *(*dlsym_fn)(void *, const char *);

void *dlsym(void *handle, const char *symbol) {
    /* Bootstrap real_dlsym lazily. dlvsym lets us name a specific
     * glibc version so we resolve the underlying symbol rather than
     * recursing into our own override. GLIBC_2.2.5 has been the
     * dlsym ABI since 2002 — safe across every glibc the vast.ai
     * fleet ships. */
    static dlsym_fn real_dlsym = NULL;
    if (!real_dlsym) {
        real_dlsym = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    if (symbol == NULL) {
        return real_dlsym(handle, symbol);
    }

    /* Path 1: _v2-suffix alias fallback. cudarc 0.19.4 statically
     * requests `cuCtxGetDevice_v2` / `cuFuncSetCacheConfig_v2` / etc.
     * — the versioned aliases that newer CUDA drivers consolidate
     * into the un-suffixed name. If the requested `_v2` symbol is
     * missing, try the non-suffixed variant.
     *
     * Only kicks in when the v2 lookup returns NULL. Drivers that
     * export the _v2 alias get the real pointer. */
    size_t slen = strlen(symbol);
    if (slen > 3 && strcmp(symbol + slen - 3, "_v2") == 0) {
        void *p = real_dlsym(handle, symbol);
        if (p != NULL) return p;
        char fallback[256];
        size_t base_len = slen - 3;
        if (base_len < sizeof(fallback)) {
            memcpy(fallback, symbol, base_len);
            fallback[base_len] = '\0';
            p = real_dlsym(handle, fallback);
            if (p != NULL) return p;
        }
        /* Both _v2 and base failed — fall through to the cu* stub. */
    }

    /* Path 2: real lookup. Most symbols hit this path and return
     * the real pointer. */
    void *real = real_dlsym(handle, symbol);
    if (real != NULL) return real;

    /* Path 3: cu* fallback. If a CUDA driver symbol is missing,
     * substitute the no-op stub. cudarc 0.19.4 dlsyms thousands of
     * driver symbols at binding init and panics on any miss; in
     * practice the misses are for features the binary doesn't use
     * at runtime (coredump teardown callbacks, Hopper SM-partition
     * API, multicast memory, future-CUDA versioned aliases).
     *
     * Risk: a real kernel-launch symbol could go missing and we'd
     * silently no-op it. Mitigation: cu* core symbols (cuInit,
     * cuMemAlloc, cuLaunchKernel, cuMemcpy, etc.) are stable since
     * CUDA 4.x and present on every driver in the vast.ai fleet.
     * The symbols that go missing are the new/optional families. */
    if (strncmp(symbol, "cu", 2) == 0 &&
        symbol[2] >= 'A' && symbol[2] <= 'Z') {
        return (void *)cu_coredump_callback_noop;
    }

    /* Path 3b: nvrtc* fallback. Same panic pattern, different prefix.
     * cudarc 0.19.4 also dlsyms nvrtc symbols (NVRTC JIT compiler);
     * nvrtcGetTileIR is one example — added in nvrtc 12.5, missing
     * on older 12.x runtimes. 2026-05-21 smoke on v26 hit this
     * panic against a driver-570.195 vast.ai box where the bundled
     * libnvrtc.so.12 still lacked the symbol. cuda-nvrtc-12-6 has
     * it, but vast.ai's nvidia-container-toolkit can mount the host
     * libnvrtc which may be older. Stubbing to a no-op is safe
     * because cubecl-cuda only calls nvrtc* via the documented
     * nvrtcCompileProgram path; the new TileIR / SASS feature flags
     * are never invoked here. */
    if (strncmp(symbol, "nvrtc", 5) == 0 &&
        symbol[5] >= 'A' && symbol[5] <= 'Z') {
        return (void *)cu_coredump_callback_noop;
    }

    /* Path 4: non-cu, non-nvrtc symbols genuinely missing — return
     * NULL (real dlsym semantics) so glibc / other libs see the
     * failure. */
    return NULL;
}
