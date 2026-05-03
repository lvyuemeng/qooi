#!/usr/bin/env bash
# Switch between OKX API profiles:  uv run python scripts/okx_profile.py live
#                                     uv run python scripts/okx_profile.py test

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/okx_profile.py <live|test>")
        sys.exit(1)

    profile = sys.argv[1]
    src = os.path.join(ROOT, f".env.{profile}")
    dst = os.path.join(ROOT, ".env")

    if not os.path.exists(src):
        print(f"Profile file not found: {src}")
        print(
            f"Available: {[f.replace('.env.', '') for f in os.listdir(ROOT) if f.startswith('.env.')]}"
        )
        sys.exit(1)

    shutil.copy2(src, dst)
    print(f"Switched to '{profile}' profile → .env")
    print()

    # Show current key (masked)
    with open(dst) as f:
        for line in f:
            if "API_KEY" in line and "=" in line:
                key = line.split("=", 1)[1].strip()
                masked = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
                print(f"  OKX_API_KEY = {masked}")
            if "FLAG" in line and "=" in line:
                flag = line.split("=", 1)[1].strip()
                print(
                    f"  OKX_FLAG   = {flag}  ({'LIVE' if flag == '0' else 'TESTNET'})"
                )


if __name__ == "__main__":
    main()
