import pandas as pd

from src.features import add_regex_features, detect_regex_signals


def test_detects_common_pii_patterns():
    text = (
        "Email me at hamza@example.com or call 07123 456 789. "
        "password=secret123 postcode SW1A 1AA card 4242 4242 4242 4242 "
        "key sk_test_abc123456789"
    )

    signals = detect_regex_signals(text)

    assert signals["has_email"] is True
    assert signals["has_phone"] is True
    assert signals["has_credit_card"] is True
    assert signals["has_api_key"] is True
    assert signals["has_password"] is True
    assert signals["has_postcode"] is True


def test_clean_text_has_no_regex_signals():
    signals = detect_regex_signals("Please summarize the meeting notes.")

    assert all(matched is False for matched in signals.values())


def test_add_regex_features_returns_numeric_columns():
    df = pd.DataFrame({"text": ["hello", "test@example.com"]})

    result = add_regex_features(df)

    assert result.loc[0, "has_email"] == 0
    assert result.loc[1, "has_email"] == 1

