from __future__ import annotations

import pytest

from xpkg.payloads import coerce_external_mapping_keys, mapping_or_empty, require_str_mapping


def test_payload_helpers_enforce_internal_string_key_policy() -> None:
    assert require_str_mapping({"labels": 1}, label="payload") == {"labels": 1}
    assert mapping_or_empty(None, label="payload.optional") == {}

    with pytest.raises(TypeError, match="payload must be a mapping"):
        require_str_mapping(["not", "a", "mapping"], label="payload")
    with pytest.raises(TypeError, match=r"payload\.optional must be a mapping"):
        mapping_or_empty(123, label="payload.optional")
    with pytest.raises(TypeError, match="payload must use string keys"):
        require_str_mapping({b"labels": 1}, label="payload")


def test_external_payload_helper_coerces_keys_once() -> None:
    payload = coerce_external_mapping_keys(
        {b"labels": {"frames": {}}, 7: "non-string", "metadata": {}},
        label="read_xpkg(...) payload",
    )

    assert payload == {
        "labels": {"frames": {}},
        "7": "non-string",
        "metadata": {},
    }
