"""Tests for ApiResponse envelope and ResponseMeta."""
from quant_engine.api.schemas.envelope import ApiResponse, ResponseMeta


def test_success_response():
    resp = ApiResponse.success({"key": "value"})
    assert resp.ok is True
    assert resp.data == {"key": "value"}
    assert resp.error is None
    assert resp.meta.cache_hit is False


def test_fail_response():
    resp = ApiResponse.fail("something broke", warnings=["w1"])
    assert resp.ok is False
    assert resp.error == "something broke"
    assert resp.meta.warnings == ["w1"]


def test_from_cached():
    resp = ApiResponse.from_cached({"cached": True}, elapsed_ms=5.0)
    assert resp.ok is True
    assert resp.meta.cache_hit is True
    assert resp.meta.elapsed_ms == 5.0


def test_meta_defaults():
    meta = ResponseMeta()
    assert meta.data_mode == "live"
    assert meta.generated_at is not None
    assert meta.warnings == []
    assert meta.cache_hit is False


def test_meta_custom_fields():
    meta = ResponseMeta(
        model_version="20260220_143022",
        predictor_type="ensemble",
        regime_suppressed=True,
    )
    assert meta.model_version == "20260220_143022"
    assert meta.predictor_type == "ensemble"
    assert meta.regime_suppressed is True


def test_serialization_roundtrip():
    resp = ApiResponse.success({"list": [1, 2, 3]}, model_version="v1")
    dumped = resp.model_dump()
    assert dumped["ok"] is True
    assert dumped["data"]["list"] == [1, 2, 3]
    assert dumped["meta"]["model_version"] == "v1"
