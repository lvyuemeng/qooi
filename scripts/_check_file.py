path = r"src\qooi\exchange\market.py"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

# Check key markers
checks = [
    ("class MarketData", "class MarketData" in c),
    ("_registry = {", "_registry = {" in c),
    ("_markets_loaded", "_markets_loaded" in c),
    ("_ensure_markets", "_ensure_markets" in c),
    ("_ob_fallback", "_ob_fallback" in c),
    ("_days_since", "_days_since" in c),
    ("OhlcvProvider", "OhlcvProvider" in c),
]
for name, found in checks:
    print(f"  {name:25s} {'YES' if found else 'NO'}")
