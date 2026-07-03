from vlm_state import STATE_KEYS, parse_state


def test_clean_json_parses_all_keys():
    raw = ('{"object": "tomato", "category": "food", "form": "sliced", '
           '"color": "red", "doneness": "raw", "texture": "wet", '
           '"description": "a freshly sliced red tomato", "confidence": 0.9}')
    out = parse_state(raw)
    assert out == {"object": "tomato", "category": "food", "form": "sliced",
                   "color": "red", "doneness": "raw", "texture": "wet",
                   "description": "a freshly sliced red tomato",
                   "confidence": 0.9}


def test_result_always_has_exactly_the_schema_keys():
    out = parse_state('{"object": "onion"}')
    assert set(out.keys()) == set(STATE_KEYS)


def test_markdown_fence_and_prose_are_stripped():
    raw = ('Sure! Here is the state:\n```json\n'
           '{"object": "egg", "form": "whole", "color": "white", '
           '"doneness": "cooking", "confidence": 0.7}\n```\nHope that helps.')
    out = parse_state(raw)
    assert out["object"] == "egg"
    assert out["doneness"] == "cooking"


def test_missing_key_becomes_none():
    out = parse_state('{"object": "pan", "color": "black"}')  # no form/doneness/confidence
    assert out["object"] == "pan"
    assert out["form"] is None
    assert out["confidence"] is None


def test_extra_keys_are_dropped():
    out = parse_state('{"object": "carrot", "temperature": "hot"}')
    assert "temperature" not in out
    assert out["object"] == "carrot"


def test_unparseable_returns_all_none():
    out = parse_state("I cannot determine the state of this image.")
    assert set(out.keys()) == set(STATE_KEYS)
    assert all(v is None for v in out.values())
