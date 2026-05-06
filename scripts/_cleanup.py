"""Cancel all pending testnet orders and print clean state."""
import os
os.environ["OKX_ENV"] = "test"

from qooi.exchange.trading import TradingClient, load_okx_env

tc = TradingClient()

print("=== Before cleanup ===")
try:
    for b in tc.balance():
        print(f"  {b.get('ccy','?'):6s} avail={b.get('availBal','?')} frozen={b.get('frozenBal','?')}")
except Exception as e:
    print(f"  balance error: {e}")

try:
    pending = tc.pending()
    print(f"  pending orders: {len(pending)}")
    for p in pending:
        oid = p.get("ordId", "")
        inst = p.get("instId", "")
        side = p.get("side", "")
        sz = p.get("sz", "")
        px = p.get("px", "")
        print(f"    {inst:15s} {side:4s} sz={sz} px={px} id={oid}")
except Exception as e:
    print(f"  pending error: {e}")

# Cancel all
print("\n=== Cancelling all orders ===")
try:
    pending = tc.pending()
    for p in pending:
        oid = p.get("ordId", "")
        inst = p.get("instId", "")
        try:
            tc.cancel(inst, oid)
            print(f"  CANCELLED {inst} id={oid}")
        except Exception as e:
            print(f"  cancel error for {inst} id={oid}: {e}")
except Exception as e:
    print(f"  pending error during cancel: {e}")

print("\n=== After cleanup ===")
try:
    for b in tc.balance():
        print(f"  {b.get('ccy','?'):6s} avail={b.get('availBal','?')} frozen={b.get('frozenBal','?')}")
except Exception as e:
    print(f"  balance error: {e}")

try:
    pending = tc.pending()
    print(f"  pending orders: {len(pending)}")
except Exception as e:
    print(f"  pending error: {e}")
