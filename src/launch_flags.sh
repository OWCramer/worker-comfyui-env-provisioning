#!/usr/bin/env bash
# Pure helper: compute the ComfyUI launch flags from environment variables.
# Sourced by start.sh and exercised directly by tests/behavioral/test_persona_image_dropper.py.

comfy_launch_flags() {
    local flags="--disable-auto-launch --verbose ${COMFY_LOG_LEVEL:-DEBUG} --log-stdout"

    # Workflow metadata in generated images (EMBED_WORKFLOW_METADATA, default on):
    # ComfyUI embeds the API-format workflow ('prompt' chunk) and UI-format graph
    # ('workflow' chunk) into saved PNGs, enabling drag-to-recreate and
    # image-to-endpoint flows. Set to "false" to strip metadata from outputs
    # (e.g. when prompts/workflows are considered sensitive).
    local embed_metadata
    embed_metadata=$(echo "${EMBED_WORKFLOW_METADATA:-true}" | tr '[:upper:]' '[:lower:]')
    case "$embed_metadata" in
        false|0|no|off)
            flags="$flags --disable-metadata"
            ;;
    esac

    if [ "${SERVE_API_LOCALLY}" = "true" ]; then
        flags="$flags --listen"
    fi

    echo "$flags"
}
