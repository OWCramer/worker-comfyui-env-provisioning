#!/usr/bin/env bash

# Start SSH server if PUBLIC_KEY is set (enables remote access and dev-sync.sh)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    # Generate host keys if they don't exist (removed during image build for security)
    for key_type in rsa ecdsa ed25519; do
        key_file="/etc/ssh/ssh_host_${key_type}_key"
        if [ ! -f "$key_file" ]; then
            ssh-keygen -t "$key_type" -f "$key_file" -q -N ''
        fi
    done

    service ssh start && echo "worker-comfyui: SSH server started" || echo "worker-comfyui: SSH server could not be started" >&2
fi

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ---------------------------------------------------------------------------
# GPU pre-flight check
# Verify that the GPU is accessible before starting ComfyUI. If PyTorch
# cannot initialize CUDA the worker will never be able to process jobs,
# so we fail fast with an actionable error message.
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    # Launch a real kernel. The driver-only calls above succeed even when this
    # PyTorch build has no compiled kernels for the GPU architecture (e.g. an
    # older torch on a newer GPU). Without this, the worker boots, ComfyUI dies
    # on the first GPU op, and it surfaces as the misleading 'server not
    # reachable' error instead of a clear cause here.
    _ = (torch.zeros(8, device='cuda') + 1).sum().item()
    torch.cuda.synchronize()
    print(f'OK: {name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}, cuda {torch.version.cuda}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
    echo "worker-comfyui: GPU is not available or incompatible with this PyTorch build:"
    echo "worker-comfyui: $GPU_CHECK"
    echo "worker-comfyui: A 'no kernel image is available' error means this torch build"
    echo "worker-comfyui: lacks kernels for this GPU. Otherwise the GPU may not be"
    echo "worker-comfyui: properly initialized — please contact RunPod support."
    exit 1
fi
echo "worker-comfyui: GPU available — $GPU_CHECK"

# ---------------------------------------------------------------------------
# Runtime provisioning + ComfyUI launch — in the BACKGROUND.
#
# Serverless platforms cull workers whose handler is not up within a health
# window (~10 minutes observed). Large model sets (e.g. 20GB+ of video
# models) cannot finish downloading inside that window, so the handler must
# start FIRST. The handler waits for ComfyUI while the provisioning marker
# exists (see handler.py); jobs received meanwhile simply wait.
#
# Provisioning failures are written to a marker the handler reports on every
# job, so users get the actual error instead of a crash-looping worker.
# ---------------------------------------------------------------------------
PROVISIONING_MARKER="/tmp/provisioning.in_progress"
PROVISIONING_FAILED="/tmp/provisioning.failed"
COMFY_PID_FILE="/tmp/comfyui.pid"

rm -f "$PROVISIONING_MARKER" "$PROVISIONING_FAILED"
touch "$PROVISIONING_MARKER"

# Launch flags (log level, metadata embedding, local API mode) are computed
# from environment variables — see src/launch_flags.sh for the contract.
source /launch_flags.sh
COMFY_FLAGS=$(comfy_launch_flags)
echo "worker-comfyui: ComfyUI launch flags: ${COMFY_FLAGS}"

(
    if ! python -u -m provisioning; then
        echo "worker-comfyui: Provisioning failed — see errors above. Fix the endpoint's environment variables." >&2
        echo "Provisioning failed at worker startup — check the endpoint's environment variables (model URLs, CUSTOM_NODES, tokens) and see worker logs for details." > "$PROVISIONING_FAILED"
        rm -f "$PROVISIONING_MARKER"
        exit 1
    fi

    # Provisioning may export PYTHONPATH for volume-cached custom node deps
    if [ -f /tmp/provision_env.sh ]; then
        source /tmp/provision_env.sh
    fi

    # Ensure ComfyUI-Manager runs in offline network mode inside the container
    comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

    echo "worker-comfyui: Starting ComfyUI"
    python -u /comfyui/main.py ${COMFY_FLAGS} &
    echo $! > "$COMFY_PID_FILE"
    rm -f "$PROVISIONING_MARKER"
) &

echo "worker-comfyui: Starting RunPod Handler (provisioning continues in background)"
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /handler.py
fi
