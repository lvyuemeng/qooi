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
                "allowMultipleEntry": allow_multiple_entry,
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

    def signal_execute_enter(self, d, algo_id: str, signal_chan_id: str, inst_id: str) -> None:
        """Push an enter signal from a Decision object.

        Encapsulates the arg decomposition so callers don't scatter it.
        """
        self.signal_push_sub_order(
            algo_id=algo_id,
            signal_chan_id=signal_chan_id,
            inst_id=inst_id,
            side=d.side,
            sz=str(int(d.sz)),
            ord_type="limit",
            px=str(d.entry_px),
        )

    def signal_execute_exit(self, algo_id: str, signal_chan_id: str, inst_id: str) -> None:
        """Close position via signal bot."""
        self.signal_close_position(
            algo_id=algo_id,
            signal_chan_id=signal_chan_id,
            inst_id=inst_id,
        )

    def signal_ensure_bot(self, pair):
        """Find or create the OKX signal bot for this pair.  Idempotent.

        Uses ``algoClOrdId`` to prevent duplicate algos — if the algo
        already exists with this client-assigned ID, OKX returns 51065
        and we re-query pending to get the existing one.

        Returns ``BotIdentity`` or ``None`` on unrecoverable failure.
        """
        from qooi.core.config import BotIdentity

        bot = self._signal_resolve_bot(pair)
        if bot is None:
            print(f"    WARNING: signal_get_pending failed for {pair.chan_name}")
            return None
        if bot.algo_id:
            return bot

        name = pair.chan_name
        desc = f"{pair.okx.strategy} signal for {pair.asset.symbol} {pair.asset.timeframe}"
        try:
            chan = self.signal_create(name, desc)
        except RuntimeError:
            chan = self._signal_find_channel(name)
            if not chan:
                print(f"    WARNING: channel exists but failed to find {pair.chan_name}")
                return None
        chan_id = (
            chan.get("signalChanId", chan.get("data", [{}])[0].get("signalChanId", ""))
            if isinstance(chan, dict)
            else ""
        )
        if not chan_id:
            print(f"    WARNING: failed to create channel for {pair.chan_name}")
            return None

        try:
            algo = self.signal_create_order_algo(
                signal_chan_id=chan_id,
                inst_ids=[pair.asset.symbol],
                lever=str(int(pair.asset.leverage)),
                invest_amt=str(int(pair.asset.capital)),
                entry_type="1",
                tp_pct=pair.okx.tp_pct,
                sl_pct=pair.okx.sl_pct,
                sub_ord_type="9",
                allow_multiple_entry=False,
                algo_cl_ord_id=f"qooi{pair.asset.symbol.replace('-', '')}v1",
            )
        except RuntimeError:
            bot = self._signal_resolve_bot(pair)
            if bot and bot.algo_id:
                return bot
            print(f"    WARNING: failed to create algo for {pair.chan_name}")
            return None

        algo_data = algo if isinstance(algo, dict) else {}
        algo_id = algo_data.get("algoId", algo_data.get("data", [{}])[0].get("algoId", ""))
        return BotIdentity(algo_id=algo_id, signal_chan_id=chan_id)

    def signal_query_position(self, bot, pair):
        """Query current position from OKX signal/positions."""
        from qooi.core.config import PositionState

        try:
            resp = self.signal_get_positions(bot.algo_id)
            for pos in resp.get("data", []) if isinstance(resp, dict) else []:
                if pos.get("instId") == pair.asset.symbol:
                    qty = str(pos.get("pos", "0"))
                    if qty != "0" and qty not in ("", "nan", "None"):
                        p = float(qty)
                        if p > 0:
                            return PositionState(has_position=True, side="buy")
                        elif p < 0:
                            return PositionState(has_position=True, side="sell")
        except Exception:
            pass
        return PositionState()

    def _signal_find_channel(self, chan_name: str) -> dict | None:
        """Find channel by name from signal/signals.

        Used when signal_create hits 60083 (name already in use) because
        a stale channel exists without an algo attached.
        """
        try:
            resp = self._retry(
                lambda: self._signal_api("GET", "signal/signals", {"signalSourceType": "1"})
            )
            for ch in resp.get("data", []) if isinstance(resp, dict) else []:
                if ch.get("signalChanName") == chan_name:
                    return ch
        except Exception:
            pass
        return None

    def _signal_resolve_bot(self, pair):
        """Find existing bot by channel name from orders-algo-pending.

        Returns:
          - ``BotIdentity(algo_id=..., signal_chan_id=...)`` if found
          - ``BotIdentity()`` if not found (caller creates new bot)
          - ``None`` on network failure (caller skips, no creation)
        """
        from qooi.core.config import BotIdentity

        try:
            resp = self.signal_get_pending()
            for bot in resp.get("data", []) if isinstance(resp, dict) else []:
                if bot.get("signalChanName") == pair.chan_name:
                    return BotIdentity(
                        algo_id=bot.get("algoId", ""),
                        signal_chan_id=bot.get("signalChanId", ""),
                    )
            return BotIdentity()
        except Exception as e:
            print(f"    WARNING: signal_get_pending failed: {e}")
            return None

    def signal_close_position(self, algo_id: str, signal_chan_id: str, inst_id: str) -> dict:
        return self._retry(
            lambda: self._signal_api(
                "POST",
                "signal/close-position",
                {"algoId": algo_id, "signalChanId": signal_chan_id, "instId": inst_id},
            )
        )

    def signal_get_pending(self) -> dict:
        return self._retry(
            lambda: self._signal_api(
                "GET", "signal/orders-algo-pending", {"algoOrdType": "contract"}
            )
        )

    def signal_get_positions(self, algo_id: str) -> dict:
        return self._retry(
            lambda: self._signal_api(
                "GET",
                "signal/positions",
                {"algoId": algo_id, "algoOrdType": "contract"},
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
        resp = self._trade._request(method, url, params)
        if isinstance(resp, dict):
            code = resp.get("code", -1)
            if str(code) != "0":
                raise RuntimeError(
                    f"OKX signal bot error [{code}]: {resp.get('msg', '')}"
                    f" data={resp.get('data', [])}"
                )
        return resp


# ============================================================================
# 2. Backtest data models
# ============================================================================
