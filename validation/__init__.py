"""
Truth Layer validation — preflight checks for the quant engine.

Modules:
    preconditions     — execution contract validation (RET_TYPE, LABEL_H, etc.)
    data_integrity    — OHLCV quality gate (blocks corrupt data)
    leakage_detection — feature causality and time-shift leakage tests
"""
