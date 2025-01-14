import sys

SUPPORTED_VERSIONS = [(3, 10), (3, 11), (3, 12)]
CURRENT_VERSION = sys.version_info[:2]

if CURRENT_VERSION not in SUPPORTED_VERSIONS:
    raise RuntimeError(
        f"ptseries-sdk requires Python 3.10, 3.11, or 3.12. "
        f"You are using Python {CURRENT_VERSION[0]}.{CURRENT_VERSION[1]}."
    )

from ptseries.common.set_seed import set_seed as set_seed
