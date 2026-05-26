#!/usr/bin/env bash
# zen-r2-lib.sh — shared R2/S3 helpers for zen sweep + fleet scripts.
#
# CANONICAL HOME: zenmetrics/scripts/lib/zen-r2-lib.sh
# (copied verbatim into jxl-encoder/scripts/zenjxl-tuning-sweep/lib/ —
#  cross-repo copy is acceptable for a sourced shell lib per the
#  dedup-Chunk-D pragmatic-first principle. A future chunk may promote
#  to a git submodule or shared install location.)
#
# Env contract (must match zenmetrics/crates/vastai-fleet/src/worker/r2.rs):
#   R2_ACCOUNT_ID        Cloudflare R2 account id (used to derive endpoint)
#   R2_ACCESS_KEY_ID     R2 access key id (re-exported as AWS_ACCESS_KEY_ID)
#   R2_SECRET_ACCESS_KEY R2 secret access key (re-exported as AWS_SECRET_ACCESS_KEY)
#
# Optional:
#   R2_ENDPOINT          override the derived endpoint
#   R2_CREDS_FILE        path to a creds file to source (default:
#                        ~/.config/cloudflare/r2-credentials).
#                        File must export R2_ACCOUNT_ID + R2_ACCESS_KEY_ID
#                        + R2_SECRET_ACCESS_KEY (the same vars vastai-fleet reads).
#
# Usage:
#   source "$(dirname "$0")/lib/zen-r2-lib.sh"
#   zen_r2_init                                # idempotent; safe to call multiple times
#   zen_r2_s3 ls "s3://zen-tuning-ephemeral/"   # aws s3 wrapper
#   zen_r2_s5 cp "s3://bucket/key" /tmp/x       # s5cmd wrapper
#   zen_r2_sync "$dir" "s3://bucket/prefix/"    # aws s3 sync (idempotent)
#   zen_r2_verify "s3://bucket/key"             # head-object check, nonzero on miss
#   zen_r2_hydrate_from_proc1environ            # vast.ai onstart pattern (10+ scripts)
#
# All helpers honour `set -euo pipefail`; do not unset on entry.

# ── internal: guard against double-init contaminating env ────────────
_ZEN_R2_LIB_INITIALIZED="${_ZEN_R2_LIB_INITIALIZED:-0}"

zen_r2_init() {
    # Idempotent. Safe to call from sourced libs that re-source us.
    if [[ "$_ZEN_R2_LIB_INITIALIZED" == "1" ]]; then
        return 0
    fi

    # 1. If R2_* not in env, source from creds file (default or override).
    local creds="${R2_CREDS_FILE:-$HOME/.config/cloudflare/r2-credentials}"
    if [[ -z "${R2_ACCOUNT_ID:-}" || -z "${R2_ACCESS_KEY_ID:-}" \
            || -z "${R2_SECRET_ACCESS_KEY:-}" ]]; then
        if [[ -r "$creds" ]]; then
            set -a
            # shellcheck disable=SC1090
            source "$creds"
            set +a
        fi
    fi

    # 2. Hard-fail with the same `: ${VAR:?}` shape every script used to inline.
    : "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing (set in env or $creds)}"
    : "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing (set in env or $creds)}"
    : "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing (set in env or $creds)}"

    # 3. Derive endpoint unless overridden.
    R2_ENDPOINT="${R2_ENDPOINT:-https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com}"

    # 4. Re-export under aws-cli names for any helper that bypasses our wrapper.
    export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"
    export R2_ENDPOINT
    export S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-$R2_ENDPOINT}"

    _ZEN_R2_LIB_INITIALIZED=1
    export _ZEN_R2_LIB_INITIALIZED
}

# ── aws s3 wrapper (replaces per-script `S3()` redefinitions) ────────
# Drop-in for the 20+ scripts that defined `S3() { aws --endpoint-url
# "$R2_ENDPOINT" "$@"; }` — keep using `S3 s3 ls ...` if you alias
# `S3=zen_r2_s3`, or just call `zen_r2_s3 s3 ls ...` directly.
zen_r2_s3() {
    zen_r2_init
    aws --endpoint-url "$R2_ENDPOINT" "$@"
}

# ── s5cmd wrapper (replaces inline `s5cmd --endpoint-url "$R2_ENDPOINT"`) ─
zen_r2_s5() {
    zen_r2_init
    s5cmd --endpoint-url "$R2_ENDPOINT" "$@"
}

# ── aws s3 sync convenience with simple retry ────────────────────────
# Usage: zen_r2_sync <src> <dst> [extra-args...]
# Retries up to 3 times on failure (network blips); --no-progress by default
# (sweep scripts already log their own progress).
zen_r2_sync() {
    local src="$1"; shift
    local dst="$1"; shift
    local extra=("$@")
    zen_r2_init
    local attempt
    for (( attempt = 1; attempt <= 3; attempt++ )); do
        if aws --endpoint-url "$R2_ENDPOINT" s3 sync "$src" "$dst" \
                --no-progress "${extra[@]}"; then
            return 0
        fi
        if (( attempt < 3 )); then
            sleep $(( attempt * 2 ))
        fi
    done
    echo "[zen-r2] sync $src → $dst failed after 3 attempts" >&2
    return 1
}

# ── head-object existence check ──────────────────────────────────────
# Returns 0 if the object exists, nonzero on miss. Mirrors the
# W44-PHASE4-S1h pre-flight pattern (jxl-encoder/scripts/zenjxl-tuning-
# sweep/launch_w44_phase4_s1_fleet.sh inlined the same idea).
#
# Usage: zen_r2_verify s3://bucket/key/path
# (accepts s3:// URI; splits to bucket + key for `aws s3api head-object`.)
zen_r2_verify() {
    local uri="$1"
    if [[ "$uri" != s3://* ]]; then
        echo "[zen-r2] verify: must be s3:// URI (got '$uri')" >&2
        return 2
    fi
    zen_r2_init
    local rest="${uri#s3://}"
    local bucket="${rest%%/*}"
    local key="${rest#*/}"
    if [[ "$key" == "$rest" ]]; then
        echo "[zen-r2] verify: URI missing key path: '$uri'" >&2
        return 2
    fi
    aws --endpoint-url "$R2_ENDPOINT" s3api head-object \
        --bucket "$bucket" --key "$key" >/dev/null 2>&1
}

# ── vast.ai onstart env-hydration pattern ────────────────────────────
# Vast.ai passes worker env via --env flags that land on PID 1's
# environ. The bash that invokes onstart sometimes inherits, sometimes
# not. Explicitly import R2_*/SWEEP_*/WORKER_*/STATS_* so onstart scripts
# don't have to inline the same 7-line snippet (currently in ~10 boot
# scripts: zenmetrics/scripts/sweep/onstart_v3.sh + 9 jxl-encoder
# onstart variants).
zen_r2_hydrate_from_proc1environ() {
    if [[ -r /proc/1/environ ]]; then
        while IFS='=' read -r -d '' k v; do
            case "$k" in
                R2_*|SWEEP_*|WORKER_*|STATS_*|W44_*) export "$k=$v" ;;
            esac
        done < /proc/1/environ
    fi
}
