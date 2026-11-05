import json

from src.model.output_validation import validate_output

IDS = {"leave-annual", "leave-sick"}


def good(**kw):
    d = {"answer": "18 days.", "source": "leave-annual", "answerable": True}
    d.update(kw)
    return json.dumps(d)


def test_valid():
    r = validate_output(good(), "leave-annual", IDS)
    assert r.valid and r.answer == "18 days." and r.source == "leave-annual"


def test_valid_refusal():
    r = validate_output(good(answer="Not in the excerpt.", source=None, answerable=False), "leave-annual", IDS)
    assert r.valid and not r.answerable


def test_code_fence_and_prose_are_tolerated():
    assert validate_output("Sure!\n```json\n" + good() + "\n```", "leave-annual", IDS).valid


def test_failures():
    assert "no_json_object" in validate_output("18 days of leave", None, IDS).errors
    assert "invalid_json" in validate_output("{answer: 18}", None, IDS).errors
    assert validate_output(json.dumps({"answer": "x"}), None, IDS).errors[0].startswith("missing_fields")
    assert "empty_or_non_string_answer" in validate_output(good(answer=" "), None, IDS).errors
    assert "answerable_not_bool" in validate_output(good(answerable="yes"), None, IDS).errors
    assert "answerable_without_source" in validate_output(good(source=None), None, IDS).errors
    assert "unknown_source_id" in validate_output(good(source="made-up"), "leave-annual", IDS).errors
    assert "source_not_in_context" in validate_output(good(source="leave-sick"), "leave-annual", IDS).errors
    assert "answer_too_long" in validate_output(good(answer="x" * 700), None, IDS).errors


def test_braces_inside_strings():
    r = validate_output(good(answer="Use {braces} carefully"), "leave-annual", IDS)
    assert r.valid
