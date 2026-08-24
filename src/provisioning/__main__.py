import sys

from .errors import ProvisionError
from .runner import provision


def main():
    try:
        provision()
    except ProvisionError as exc:
        print(
            f"worker-comfyui (provisioning): PROVISIONING FAILED\n{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
