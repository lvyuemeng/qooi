"""OKX testnet (demo) paper-trading adapter.

Uses the OKX Python SDK with flag='1' for the demo environment.
Place limit orders at best bid/ask with post_only to guarantee maker fees.

Usage::

    adapter = OkxPaperTrader(api_key="...", secret="...", passphrase="...")
    adapter.place_limit("BTC-USDT-SWAP", "buy", sz=0.001, px=50000)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeLog:
    """Immutable record of a placed order."""

    inst_id: str
    side: str
    sz: float
    px: float
    ord_type: str = "limit"
    td_mode: str = "isolated"
    status: str = "placed"


class OkxPaperTrader:
    """Place limit orders and track P&L on OKX demo (testnet).

    ``flag='1'`` selects the demo environment.  No real funds are used.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        flag: str = "1",
    ) -> None:
        from okx.Trade import TradeAPI

        self._api = TradeAPI(api_key, secret_key, passphrase, flag=flag, debug=False)
        self._log: list[TradeLog] = []

    def market_buy(
        self,
        inst_id: str,
        sz: float,
        td_mode: str = "isolated",
    ) -> dict:
        return self._place(inst_id, "buy", sz, px="", ord_type="market", td_mode=td_mode)

    def market_sell(
        self,
        inst_id: str,
        sz: float,
        td_mode: str = "isolated",
    ) -> dict:
        return self._place(inst_id, "sell", sz, px="", ord_type="market", td_mode=td_mode)

    def limit_buy(
        self,
        inst_id: str,
        sz: float,
        px: float,
        td_mode: str = "isolated",
    ) -> dict:
        return self._place(inst_id, "buy", sz, px=str(px), ord_type="post_only", td_mode=td_mode)

    def limit_sell(
        self,
        inst_id: str,
        sz: float,
        px: float,
        td_mode: str = "isolated",
    ) -> dict:
        return self._place(inst_id, "sell", sz, px=str(px), ord_type="post_only", td_mode=td_mode)

    def _place(
        self, inst_id: str, side: str, sz: float, px: str, ord_type: str, td_mode: str
    ) -> dict:
        params = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": ord_type,
            "sz": str(sz),
        }
        if ord_type != "market":
            params["px"] = px
        resp = self._api.place_order(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX order error: {resp}")
        self._log.append(
            TradeLog(
                inst_id=inst_id,
                side=side,
                sz=sz,
                px=float(px) if px else 0.0,
                ord_type=ord_type,
                td_mode=td_mode,
            )
        )
        return resp

    def get_balance(self, ccy: str | None = None) -> dict:
        from okx.Account import AccountAPI

        acc = AccountAPI(
            self._api.api_key,
            self._api.secret_key,
            self._api.passphrase,
            flag="1",
            debug=False,
        )
        resp = acc.get_account_balance(ccy=ccy)
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX balance error: {resp}")
        return resp

    def get_positions(self) -> dict:
        from okx.Account import AccountAPI

        acc = AccountAPI(
            self._api.api_key,
            self._api.secret_key,
            self._api.passphrase,
            flag="1",
            debug=False,
        )
        resp = acc.get_positions()
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX positions error: {resp}")
        return resp

    @property
    def log(self) -> list[TradeLog]:
        return list(self._log)

    def clear_log(self) -> None:
        self._log.clear()
