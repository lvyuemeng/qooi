"""OKX trading — place/cancel/amend orders, account/position queries.

Requires API key set as environment variables (never hardcoded):

    export OKX_API_KEY="your-api-key"
    export OKX_SECRET_KEY="your-secret-key"
    export OKX_PASSPHRASE="your-passphrase"

Or create a ``.env`` file in the project root:

    OKX_API_KEY=your-api-key
    OKX_SECRET_KEY=your-secret-key
    OKX_PASSPHRASE=your-passphrase
"""

from __future__ import annotations

import os

import polars as pl
from dotenv import load_dotenv
from okx.Account import AccountAPI
from okx.Trade import TradeAPI

load_dotenv()


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(
            f"Missing {key} environment variable. "
            f"Set it via:\n"
            f"    export {key}='your-value'\n"
            f"Or create a .env file with:\n"
            f"    {key}=your-value"
        )
    return val


class TradingClient:
    """OKX trading operations — orders, account, positions.

    Usage::

        tc = TradingClient()
        tc.place_order("BTC-USDT", side="buy", ord_type="market", sz="0.01")
        bal = tc.balance()
        pos = tc.positions()
    """

    def __init__(self, flag: str = "1") -> None:
        """
        Parameters
        ----------
        flag:
            ``"0"`` for live trading, ``"1"`` for demo (testnet — recommended first).
        """
        api_key = _require_env("OKX_API_KEY")
        secret = _require_env("OKX_SECRET_KEY")
        phrase = _require_env("OKX_PASSPHRASE")

        self._trade = TradeAPI(api_key, secret, phrase, flag=flag, debug=False)
        self._account = AccountAPI(api_key, secret, phrase, flag=flag, debug=False)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        inst_id: str,
        side: str,
        ord_type: str = "market",
        sz: str = "",
        px: str | None = None,
        td_mode: str = "cash",
    ) -> dict:
        """Place an order.

        Parameters
        ----------
        inst_id:
            e.g. ``"BTC-USDT"``
        side:
            ``"buy"`` or ``"sell"``
        ord_type:
            ``"market"``, ``"limit"``, ``"post_only"``, ``"fok"``, ``"ioc"``
        sz:
            Amount to buy/sell (in base currency for spot).
        px:
            Limit price (required for limit orders).
        td_mode:
            ``"cash"`` for spot, ``"cross"`` / ``"isolated"`` for margin/derivatives.
        """
        params = {
            "instId": inst_id,
            "side": side,
            "ordType": ord_type,
            "sz": sz,
            "tdMode": td_mode,
        }
        if px:
            params["px"] = px

        resp = self._trade.place_order(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"Place order failed: {resp.get('msg', resp)}")
        return resp["data"][0] if resp.get("data") else resp

    def cancel_order(
        self, inst_id: str, ord_id: str | None = None, cl_ord_id: str | None = None
    ) -> dict:
        """Cancel an order by order ID or client order ID."""
        params = {"instId": inst_id}
        if ord_id:
            params["ordId"] = ord_id
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        resp = self._trade.cancel_order(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"Cancel order failed: {resp.get('msg', resp)}")
        return resp["data"][0] if resp.get("data") else resp

    def amend_order(
        self,
        inst_id: str,
        ord_id: str | None = None,
        cl_ord_id: str | None = None,
        new_px: str | None = None,
        new_sz: str | None = None,
    ) -> dict:
        """Amend an existing order (price and/or size)."""
        params = {"instId": inst_id}
        if ord_id:
            params["ordId"] = ord_id
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        if new_px:
            params["newPx"] = new_px
        if new_sz:
            params["newSz"] = new_sz
        resp = self._trade.amend_order(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"Amend order failed: {resp.get('msg', resp)}")
        return resp["data"][0] if resp.get("data") else resp

    def get_order(
        self, inst_id: str, ord_id: str | None = None, cl_ord_id: str | None = None
    ) -> pl.DataFrame:
        """Get order details."""
        params = {"instId": inst_id}
        if ord_id:
            params["ordId"] = ord_id
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        resp = self._trade.get_order(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"Get order failed: {resp.get('msg', resp)}")
        return pl.DataFrame(resp.get("data", []))

    def pending_orders(self, inst_type: str | None = None) -> pl.DataFrame:
        """List all pending/open orders."""
        params = {}
        if inst_type:
            params["instType"] = inst_type
        resp = self._trade.get_order_list(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"Get pending orders failed: {resp.get('msg', resp)}")
        return pl.DataFrame(resp.get("data", []))

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def balance(self, ccy: str | None = None) -> pl.DataFrame:
        """Get account balance for all currencies (or a specific one)."""
        params = {}
        if ccy:
            params["ccy"] = ccy
        resp = self._account.get_account_balance(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"Get balance failed: {resp.get('msg', resp)}")
        details = resp.get("data", [{}])[0].get("details", [])
        return pl.DataFrame(details)

    def positions(
        self, inst_type: str | None = None, inst_id: str | None = None
    ) -> pl.DataFrame:
        """Get current positions."""
        params = {}
        if inst_type:
            params["instType"] = inst_type
        if inst_id:
            params["instId"] = inst_id
        resp = self._account.get_positions(**params)
        if resp.get("code") != "0":
            raise RuntimeError(f"Get positions failed: {resp.get('msg', resp)}")
        return pl.DataFrame(resp.get("data", []))
