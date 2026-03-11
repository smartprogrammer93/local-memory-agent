#!/usr/bin/env bash
set -euo pipefail

# install_qmd_wrapper.sh — Install local-memory-agent-cli CLI wrapper for OpenClaw memory backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="$SCRIPT_DIR/qmd_wrapper.py"
DEFAULT_PREFIX="$HOME/.local/bin"
OPENCLAW_CONFIG="$HOME/.openclaw/openclaw.json"
OPENCLAW_BACKUP="$HOME/.openclaw/openclaw.json.bak"

# ── Parse arguments ──────────────────────────────────────────
ACTION="install"
PREFIX="$DEFAULT_PREFIX"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)   ACTION="apply";   shift ;;
        --restore) ACTION="restore"; shift ;;
        --prefix)  PREFIX="$2";      shift 2 ;;
        --prefix=*) PREFIX="${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--apply | --restore] [--prefix DIR]"
            echo ""
            echo "  (default)   Install local-memory-agent-cli to ~/.local/bin and print config instructions"
            echo "  --apply     Also patch ~/.openclaw/openclaw.json (backs up to .bak)"
            echo "  --restore   Restore openclaw.json from .bak, or remove memory.qmd.command"
            echo "  --prefix    Install directory (default: ~/.local/bin)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DEST="$PREFIX/local-memory-agent-cli"

# ── Helpers ──────────────────────────────────────────────────

check_prerequisites() {
    # Validate source exists
    if [[ ! -f "$SOURCE_FILE" ]]; then
        echo "ERROR: qmd_wrapper.py not found at $SOURCE_FILE" >&2
        echo "Run this script from the local-memory-agent directory." >&2
        exit 1
    fi

    # Validate python3
    if ! command -v python3 &>/dev/null; then
        echo "WARNING: python3 not found in PATH. local-memory-agent-cli requires python3." >&2
    else
        # Validate requests module
        if ! python3 -c "import requests" 2>/dev/null; then
            echo "WARNING: python3 'requests' module not found. Install with: pip3 install requests" >&2
        fi
    fi
}

do_install() {
    check_prerequisites

    # Ensure target directory exists
    mkdir -p "$PREFIX"

    # Copy wrapper
    cp "$SOURCE_FILE" "$DEST"

    # Ensure shebang is present (replace first line if needed)
    if ! head -1 "$DEST" | grep -q '^#!/usr/bin/env python3'; then
        sed -i '1s|^.*$|#!/usr/bin/env python3|' "$DEST"
    fi

    chmod +x "$DEST"

    echo "Installed local-memory-agent-cli to $DEST"
    echo ""
    echo "To configure OpenClaw, add to $OPENCLAW_CONFIG:"
    echo ""
    echo '  "memory": { "backend": "qmd", "qmd": { "command": "'"$DEST"'" } }'
    echo ""
    echo "Or run: $0 --apply"
}

# ── JSON patching ────────────────────────────────────────────

patch_json_jq() {
    local file="$1" dest_path="$2"
    jq --arg cmd "$dest_path" '.memory = (.memory // {}) | .memory.backend = "qmd" | .memory.qmd = (.memory.qmd // {}) | .memory.qmd.command = $cmd' "$file"
}

patch_json_python() {
    local file="$1" dest_path="$2"
    python3 -c "
import json, sys
with open('$file') as f:
    cfg = json.load(f)
mem = cfg.setdefault('memory', {})
mem['backend'] = 'qmd'
qmd = mem.setdefault('qmd', {})
qmd['command'] = '$dest_path'
print(json.dumps(cfg, indent=2))
"
}

remove_qmd_command_jq() {
    local file="$1"
    jq 'if .memory.qmd.command then del(.memory.qmd.command) else . end | if .memory.qmd == {} then del(.memory.qmd) else . end' "$file"
}

remove_qmd_command_python() {
    local file="$1"
    python3 -c "
import json
with open('$file') as f:
    cfg = json.load(f)
if 'memory' in cfg and 'qmd' in cfg['memory']:
    cfg['memory']['qmd'].pop('command', None)
    if not cfg['memory']['qmd']:
        del cfg['memory']['qmd']
print(json.dumps(cfg, indent=2))
"
}

do_apply() {
    # Install first
    do_install

    if [[ ! -f "$OPENCLAW_CONFIG" ]]; then
        echo "WARNING: $OPENCLAW_CONFIG not found. Creating minimal config." >&2
        mkdir -p "$(dirname "$OPENCLAW_CONFIG")"
        echo '{}' > "$OPENCLAW_CONFIG"
    fi

    # Backup
    cp "$OPENCLAW_CONFIG" "$OPENCLAW_BACKUP"
    echo "Backed up config to $OPENCLAW_BACKUP"

    # Patch
    local patched
    if command -v jq &>/dev/null; then
        patched="$(patch_json_jq "$OPENCLAW_CONFIG" "$DEST")"
    else
        patched="$(patch_json_python "$OPENCLAW_CONFIG" "$DEST")"
    fi

    echo "$patched" > "$OPENCLAW_CONFIG"
    echo "Patched $OPENCLAW_CONFIG with memory.qmd.command = $DEST"
}

do_restore() {
    if [[ -f "$OPENCLAW_BACKUP" ]]; then
        cp "$OPENCLAW_BACKUP" "$OPENCLAW_CONFIG"
        echo "Restored $OPENCLAW_CONFIG from $OPENCLAW_BACKUP"
    elif [[ -f "$OPENCLAW_CONFIG" ]]; then
        # No backup — just remove the memory.qmd.command field
        local patched
        if command -v jq &>/dev/null; then
            patched="$(remove_qmd_command_jq "$OPENCLAW_CONFIG")"
        else
            patched="$(remove_qmd_command_python "$OPENCLAW_CONFIG")"
        fi
        echo "$patched" > "$OPENCLAW_CONFIG"
        echo "Removed memory.qmd.command from $OPENCLAW_CONFIG (no .bak found)"
    else
        echo "Nothing to restore: neither $OPENCLAW_CONFIG nor $OPENCLAW_BACKUP exist."
    fi
}

# ── Main ─────────────────────────────────────────────────────

case "$ACTION" in
    install) do_install ;;
    apply)   do_apply ;;
    restore) do_restore ;;
esac
