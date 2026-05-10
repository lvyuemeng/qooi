"""Apply all patches to trading.py via Python script (exec_shell-safe)."""
import textwrap

path = r"src\qooi\exchange\trading.py"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

print(f"Read {len(c)} bytes")

# ---- 1. Fix __init__ env loading ----
c = c.replace(
    '        if not os.getenv("OKX_API_KEY") and not os.getenv("OKX_API_KEY_TEST"):\n            load_okx_env()',
    '        if os.getenv("OKX_ENV"):\n            load_okx_env()\n        elif not os.getenv("OKX_API_KEY") and not os.getenv("OKX_API_KEY_TEST"):\n            load_okx_env()'
)

# ---- 2. Add cl_ord_id param to place() ----
c = c.replace(
    '        td_mode: str = "isolated",\n    ) -> dict:',
    '        td_mode: str = "isolated",\n        cl_ord_id: str = "",\n    ) -> dict:'
)
c = c.replace(
    '        if px:\n            params["px"] = px\n        return self._okx(self._trade.place_order(**params))',
    '        if px:\n            params["px"] = px\n        if cl_ord_id:\n            params["clOrdId"] = cl_ord_id\n        return self._okx(self._trade.place_order(**params))'
)

# ---- 3. Add new trading methods (4-space indent for class body) ----
new_trade = textwrap.indent(textwrap.dedent("""\
def order_by_cloid(self, inst_id, cl_ord_id):
    try:
        return self._okx(self._trade.get_order(instId=inst_id, clOrdId=cl_ord_id))
    except RuntimeError as e:
        if "51603" in str(e) or "order does not exist" in str(e).lower():
            return None
        raise

def place_algo(self, inst_id, side, sz, ord_type, td_mode="isolated", *,
               sl_trigger_px="", sl_ord_px="-1", tp_trigger_px="", tp_ord_px="-1"):
    params = {"instId": inst_id, "tdMode": td_mode, "side": side, "sz": sz, "ordType": ord_type}
    if sl_trigger_px:
        params["slTriggerPx"] = sl_trigger_px
        params["slOrdPx"] = sl_ord_px
    if tp_trigger_px:
        params["tpTriggerPx"] = tp_trigger_px
        params["tpOrdPx"] = tp_ord_px
    return self._okx(self._trade.place_algo_order(**params))

def cancel_algo(self, inst_id, algo_id):
    return self._okx(self._trade.cancel_algo_order(instId=inst_id, algoId=algo_id))

def pending_algo(self, inst_id="", ord_type=""):
    params = {}
    if inst_id:
        params["instId"] = inst_id
    if ord_type:
        params["ordType"] = ord_type
    return self._okx(self._trade.get_algo_order_list(**params)).get("data", [])
"""), "    ")
balance_marker = '    def balance(self, ccy: str | None = None) -> list:'
c = c.replace(balance_marker, new_trade + "\n" + balance_marker)

# ---- 4. Add account methods ----
new_acct = textwrap.indent(textwrap.dedent("""\
def close_position(self, inst_id, mgn_mode="isolated"):
    return self._okx(self._trade.close_position(instId=inst_id, mgnMode=mgn_mode))

def set_leverage(self, inst_id, lever, mgn_mode="isolated"):
    return self._okx(self._account.set_leverage(instId=inst_id, lever=str(lever), mgnMode=mgn_mode))

def account_config(self):
    return self._okx(self._account.get_account_config())
"""), "    ")
c = c.replace(balance_marker, new_acct + "\n" + balance_marker)

# ---- 5. Add _last_signal_ts ----
c = c.replace(
    "        self._signal_threshold: float = 0.40  # default, overridden by signal",
    "        self._signal_threshold: float = 0.40\n        self._last_signal_ts: int = 0"
)

# ---- 6. Update step() ----
c = c.replace(
    "        if sr.threshold > 0:\n            self._signal_threshold = sr.threshold",
    "        if sr.threshold > 0:\n            self._signal_threshold = sr.threshold\n        if sr.timestamp > 0:\n            self._last_signal_ts = sr.timestamp"
)

# ---- 7. Add cl_ord_id to OrderPayload ----
c = c.replace(
    "    ofi_flow: float = 0.0",
    '    ofi_flow: float = 0.0\n    cl_ord_id: str = ""'
)

# ---- 8. Update _place() for clOrdId ----
c = c.replace(
    '            status="placed",\n        )',
    '            status="placed",\n            cl_ord_id=str(self._last_signal_ts) if self._last_signal_ts else "",\n        )'
)
c = c.replace(
    '                    ord_type=otype, px=str(px), td_mode=self._td_mode,\n                )',
    '                    ord_type=otype, px=str(px), td_mode=self._td_mode,\n                    cl_ord_id=str(self._last_signal_ts) if self._last_signal_ts else "",\n                )'
)

with open(path, "w", encoding="utf-8") as f:
    f.write(c)
print(f"Written {len(c)} bytes")

# ---- Verify ----
import re
methods = re.findall(r'^\s{4}def (\w+)\(', c, re.MULTILINE)
print(f"Methods in class: {methods}")
print(f"place_algo: {'place_algo' in methods}")
print(f"close_position: {'close_position' in methods}")
