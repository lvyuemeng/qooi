from __future__ import annotations

from qooi.core.basket import BasketBook


def test_basket_engine_imports_public_book():
    book = BasketBook()

    assert book.active() == []
