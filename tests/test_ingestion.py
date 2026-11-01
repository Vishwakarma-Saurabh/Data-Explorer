
import pytest
from src.ingestion import read_csv_with_fallback


def test_comma_csv():
    raw = b"a,b\n1,x\n2,y\n"
    df, strategy = read_csv_with_fallback(raw)
    assert df.shape == (2, 2)
    assert "comma" in strategy


def test_semicolon_csv():
    raw = b"a;b\n1;x\n2;y\n"
    df, strategy = read_csv_with_fallback(raw)
    assert df.shape[1] == 2, "Semicolon CSV must not collapse to 1 column"


def test_empty_csv_raises():
    with pytest.raises(ValueError):
        read_csv_with_fallback(b"")