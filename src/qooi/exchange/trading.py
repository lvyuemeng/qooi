"""OKX trading client — thin SDK wrapper + signal bot endpoints.

All strategy logic lives in qooi.core.  This file is pure I/O.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# ============================================================================
# 0. Environment
# ============================================================================


def load_okx_env(env_path: str | None = None) -> None:
    path = None
    if env_path:
        path = Path(env_path)
    elif os.getenv("OKX_ENV"):
        inferred = (
            Path(__file__).resolve().parent.parent.parent.parent / f".env.{os.getenv('OKX_ENV')}"
        )
        if inferred.exists():
            path = inferred
    else:
        default = Path(".env")
        if default.exists():
            path = default
    load_dotenv(path, override=True) if path else load_dotenv(override=True)


# ============================================================================
# 1. Trading client
# ============================================================================


class TradingClient:
    _RETRY_ATTEMPTS = 3
    _RETRY_DELAY = 1.0

    def __init__(self) -> None:
        from okx.Account import AccountAPI
        from okx.Trade import TradeAPI

        if os.getenv("OKX_ENV"):
            load_okx_env()
        elif not os.getenv("OKX_API_KEY") and not os.getenv("OKX_API_KEY_TEST"):
            load_okx_env()
        flag = os.getenv("OKX_FLAG", "1")
        k = os.getenv("OKX_API_KEY") or os.getenv("OKX_API_KEY_TEST", "")
        s = os.getenv("OKX_SECRET_KEY") or os.getenv("OKX_SECRET_KEY_TEST", "")
        p = os.getenv("OKX_PASSPHRASE") or os.getenv("OKX_PASSPHRASE_TEST", "")
        if not k:
            raise RuntimeError("Missing OKX_API_KEY — call load_okx_env() or set envvars")
        self._trade = TradeAPI(k, s, p, flag=flag, debug=False)
        self._account = AccountAPI(k, s, p, flag=flag, debug=False)

    @staticmethod
    def _okx(resp: dict, key: str = "data") -> dict:
        if resp.get("code") != "0":
            code = resp.get("code", "?")
            msg = resp.get("msg", str(resp))
            data = resp.get("data", [])
            raise RuntimeError(f"OKX error [{code}]: {msg}  data={data}")
        return resp.get(key, [{}])[0] if resp.get(key) else {}

    @staticmethod
    def _retry(fn, *args, **kwargs):
        attempts = TradingClient._RETRY_ATTEMPTS
        last_err = None
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < attempts - 1:
                    time.sleep(TradingClient._RETRY_DELAY * (attempt + 1))
        if last_err is not None:
            raise last_err

    # -- order operations -------------------------------------------------------

    def place(
        self,
        inst_id: str,
        side: str,
        sz: str,
        ord_type: str = "post_only",
        px: str | None = None,
        td_mode: str = "isolated",
        cl_ord_id: str = "",
        attach_algo_ords: list[dict] | None = None,
    ) -> dict:
        params = {"instId": inst_id, "side": side, "ordType": ord_type, "sz": sz, "tdMode": td_mode}
        if px:
            params["px"] = px
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        if attach_algo_ords:
            params["attachAlgoOrds"] = attach_algo_ords
        return self._okx(self._trade.place_order(**params))

    def cancel(self, inst_id: str, ord_id: str) -> dict:
        return self._okx(self._trade.cancel_order(instId=inst_id, ordId=ord_id))

    def set_leverage(self, inst_id, lever, mgn_mode="isolated"):
        return self._okx(
            self._account.set_leverage(instId=inst_id, lever=str(lever), mgnMode=mgn_mode)
        )

    def balance(self, ccy: str | None = None) -> list:
        p = {} if not ccy else {"ccy": ccy}
        return self._okx(self._account.get_account_balance(**p)).get("details", [])

    # -- signal bot (tradingBot endpoints, not in python-okx SDK) ---------------

    def signal_create(self, signal_chan_name: str, signal_chan_desc: str = "") -> dict:
        params = {"signalChanName": signal_chan_name}
        if signal_chan_desc:
            params["signalChanDesc"] = signal_chan_desc
        return self._retry(lambda: self._signal_api("POST", "signal/create-signal", params))

    def signal_create_order_algo(
        self,
        signal_chan_id: str,
        inst_ids: list[str],
        lever: str = "2",
        invest_amt: str = "",
        entry_type: str = "3",
        amt: str = "",
        tp_pct: str = "",
        sl_pct: str = "",
        sub_ord_type: str = "9",
        allow_multiple_entry: bool = False,
        algo_cl_ord_id: str = "",
    ) -> dict:
        params: dict = {
            "signalChanId": signal_chan_id,
            "instIds": inst_ids,
            "algoOrdType": "contract",
            "lever": lever,
            "subOrdType": sub_ord_type,
            "entrySettingParam": {
                "allowMultipleEntry": "true" if allow_multiple_entry else "false",
                "entryType": entry_type,
                "amt": amt or "",
            },
            "exitSettingParam": {"tpSlType": "price", "tpPct": tp_pct, "slPct": sl_pct},
        }
        if invest_amt:
            params["investAmt"] = invest_amt
        if algo_cl_ord_id:
            params["algoClOrdId"] = algo_cl_ord_id
        return self._retry(lambda: self._signal_api("POST", "signal/order-algo", params))

    def signal_push_sub_order(
        self,
        algo_id: str,
        signal_chan_id: str,
        inst_id: str,
        side: str,
        sz: str,
        ord_type: str = "limit",
        px: str = "",
        attach_algo_ords: list[dict] | None = None,
    ) -> dict:
        params: dict = {
            "algoId": algo_id,
            "signalChanId": signal_chan_id,
            "instId": inst_id,
            "side": side,
            "sz": sz,
            "ordType": ord_type,
        }
        if px:
            params["px"] = px
        if attach_algo_ords:
            params["attachAlgoOrds"] = attach_algo_ords
        return self._retry(lambda: self._signal_api("POST", "signal/sub-order", params))

    def signal_get_details(self, algo_id: str) -> dict:
        return self._retry(
            lambda: self._signal_api(
                "GET", "signal/orders-algo-details", {"algoId": algo_id, "algoOrdType": "contract"}
            )
        )

    def signal_close_position(self, algo_id: str, signal_chan_id: str, inst_id: str) -> dict:
        return self._retry(
            lambda: self._signal_api(
                "POST",
                "signal/close-position",
                {"algoId": algo_id, "signalChanId": signal_chan_id, "instId": inst_id},
            )
        )

    def signal_stop(self, algo_id: str, signal_chan_id: str) -> dict:
        return self._retry(
            lambda: self._signal_api(
                "POST", "signal/cancel-algo", {"algoId": algo_id, "signalChanId": signal_chan_id}
            )
        )

    def _signal_api(self, method: str, path: str, params: dict) -> dict:
        url = f"/api/v5/tradingBot/{path}"
        if method == "GET":
            resp = self._trade.get(url=url, params=params)
        else:
            resp = self._trade._request_with_params("POST", url, params)
        if isinstance(resp, dict):
            code = resp.get("code", -1)
            if str(code) != "0":
                raise RuntimeError(
                    f"OKX signal bot error [{code}]: {resp.get('msg', '')}"
                    f" data={resp.get('data', [])}"
                )
        return resp


from qooi.exchange.types import (  # noqa: F401
    FillStatus,
    OrderPayload,
    PositionState,
    State,
)


# ============================================================================
# 2. Backtest data models — re-exports from qooi.exchange.types
# ============================================================================
