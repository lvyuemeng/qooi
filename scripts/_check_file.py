path = r"src\qooi\exchange\market.py"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()
print("Lazy CCXT:", "_markets_loaded" in c)
print("_days_since removed:", "_days_since" not in c)
print("registry:", "_registry" in c)
print("Size:", len(c))
