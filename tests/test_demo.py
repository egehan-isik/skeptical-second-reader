"""Tests for the demo's logic and, through Streamlit's test runner, for its page."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from skeptical_second_reader import demo
from skeptical_second_reader import pipeline as pl
from skeptical_second_reader import second_reader as sr

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
ANSWER = (
    '{"decision": "suspicious", "confidence": 0.7, "evidence": [1], '
    '"reason": "because"}'
)
REPLY = {"content": ANSWER, "prompt_tokens": 1, "answer_tokens": 1, "seconds": 2.0}


SUMMARY = (
    "ACCOUNT A, 1-10 September: 37 transactions: 8 sent, 27 received, 2 to itself, of "
    "which 1 currency conversions. 16 counterparties. Sent 10445803.67 USD, received "
    "200475.10 USD. Formats: ACH 25, Cash 2, Cheque 9, Reinvestment 1. "
    "Showing the 30 largest of 37 transactions."
)


def prompt_of(condition):
    evidence = (
        f"{SUMMARY}"
        + chr(10)
        + "number,time,direction,counterparty,format,amount_usd,paid_currency,"
        "received_currency,transactions_in_this_direction_with_this_counterparty\n"
        "1,Sep-02 09:00,sent,K1,Credit Card,10.00,US Dollar,US Dollar,1"
    )
    ml = (
        None if condition == sr.K1 else f"UPSTREAM ML MODEL OUTPUT\nLevel: {condition}."
    )
    return sr.build_prompt(evidence, ml)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    rows = []
    for account, level in [("1:A", sr.STRONG), ("2:B", sr.NOT_FLAGGED)]:
        for condition in sr.CONDITIONS:
            rows.append(
                {
                    "account": account,
                    "group": "g",
                    "condition": condition,
                    "prompt": prompt_of(condition) + account[:0],
                    "shown_level": None if condition == sr.K1 else level,
                    "shown_rate": np.nan if condition == sr.K1 else 0.5,
                    "flipped": None if condition == sr.K1 else False,
                }
            )
    pd.DataFrame(rows).to_parquet(tmp_path / demo.PROMPTS)
    pd.DataFrame({"label": [1, 0]}, index=["1:A", "2:B"]).to_parquet(
        tmp_path / demo.READING
    )
    monkeypatch.setenv(demo.DIRECTORY_VARIABLE, str(tmp_path))
    return tmp_path


def store_answer(folder, prompt, model="gemma4:latest"):
    cache = pl.AnswerCache(folder / demo.ANSWERS)
    digest = f"version of {model}"
    cache.put(pl.answer_key(prompt, digest, False), model, digest, REPLY)
    cache.close()


def test_the_folder_can_be_chosen(folder):
    assert demo.data_directory() == folder


def test_cases_have_names_and_levels_and_no_labels(folder):
    cases = demo.load_cases(folder)
    assert cases["case"].tolist() == ["Case 001", "Case 002"]
    levels = cases.set_index("account")["level"]
    assert levels["1:A"] == sr.STRONG and levels["2:B"] == sr.NOT_FLAGGED
    assert "group" not in cases.columns and "label" not in cases.columns


def test_the_case_order_does_not_follow_the_sample_and_does_not_change(tmp_path):
    accounts = [f"a{i}" for i in range(50)]
    pd.DataFrame(
        {"account": accounts, "condition": sr.K2, "shown_level": sr.STRONG}
    ).to_parquet(tmp_path / demo.PROMPTS)
    first = demo.load_cases(tmp_path)
    assert first["account"].tolist() != accounts
    assert sorted(first["account"]) == sorted(accounts)
    pd.testing.assert_frame_equal(first, demo.load_cases(tmp_path))


def test_a_prompt_splits_into_its_parts():
    parts = demo.split_prompt(prompt_of(sr.K2))
    assert parts["summary"].startswith("ACCOUNT A")
    assert parts["transactions"]["format"].tolist() == ["Credit Card"]
    assert parts["ml_text"].startswith("UPSTREAM")
    assert demo.split_prompt(prompt_of(sr.K1))["ml_text"] is None


def test_the_summary_is_parsed():
    parsed = demo.parse_summary(SUMMARY)
    assert parsed["transactions"] == 37
    assert (parsed["sent"], parsed["received"], parsed["to_itself"]) == (8, 27, 2)
    assert (parsed["conversions"], parsed["counterparties"]) == (1, 16)
    assert parsed["sent_usd"] == 10445803.67 and parsed["received_usd"] == 200475.10
    assert parsed["formats"] == "ACH 25, Cash 2, Cheque 9, Reinvestment 1"
    assert parsed["shown"] == 30


def test_a_short_account_shows_all_its_transactions():
    text = SUMMARY.replace(" Showing the 30 largest of 37 transactions.", "")
    assert demo.parse_summary(text)["shown"] == 37


def test_the_real_summary_form_is_parsed():
    rows = pd.DataFrame(
        {
            "sender": ["1:A", "2:B"],
            "receiver": ["2:B", "1:A"],
            "timestamp": pd.to_datetime(["2022-09-01 10:00", "2022-09-01 11:00"]),
            "payment_format": ["ACH", "Wire"],
            "paid_currency": ["USD", "USD"],
            "received_currency": ["USD", "USD"],
            "usd": [10.0, 20.0],
            "usd_received": [10.0, 20.0],
            "row": [0, 1],
            "pair_txs": [1, 1],
        }
    )
    evidence, _ = sr.account_evidence(rows, "1:A")
    parsed = demo.parse_summary(evidence.split(chr(10))[0])
    assert parsed["sent"] == 1 and parsed["received_usd"] == 20.0


def test_an_unknown_summary_is_refused():
    with pytest.raises(ValueError, match="unexpected form"):
        demo.parse_summary("something else")


def test_two_models_are_compared_only_when_both_answered():
    assert demo.models_comparison([sr.SUSPICIOUS, None]) is None
    assert "same" in demo.models_comparison([sr.SUSPICIOUS, sr.SUSPICIOUS])
    assert "different" in demo.models_comparison([sr.SUSPICIOUS, sr.NOT_SUSPICIOUS])


def test_every_level_has_a_colour_and_a_mark():
    for level in [sr.STRONG, sr.UNCERTAIN, sr.NOT_FLAGGED]:
        assert level in demo.LEVEL_COLORS and level in demo.LEVEL_MARKS


def test_the_ml_details_leave_out_the_heading_and_the_level():
    text = chr(10).join(["UPSTREAM ML MODEL OUTPUT", "Level: STRONG ALERT.", "Rate."])
    assert demo.ml_details(text) == "Rate."


def test_a_missing_prompt_is_refused(folder):
    prompts = pd.read_parquet(folder / demo.PROMPTS)
    assert demo.case_prompt(prompts, "1:A", sr.K2)["shown_level"] == sr.STRONG
    with pytest.raises(ValueError, match="expected one"):
        demo.case_prompt(prompts, "nobody", sr.K2)


def test_a_stored_answer_is_found_without_the_local_model(folder):
    prompt = prompt_of(sr.K2)
    assert demo.cached_opinion(folder, prompt, "gemma4:latest") is None
    store_answer(folder, prompt)
    opinion = demo.cached_opinion(folder, prompt, "gemma4:latest")
    assert opinion["decision"] == sr.SUSPICIOUS and opinion["seconds"] == 2.0
    assert demo.cached_opinion(folder, prompt + "x", "gemma4:latest") is None
    assert demo.cached_opinion(folder, prompt, "another model") is None


def test_a_live_answer_is_kept_and_asked_once(folder, monkeypatch):
    monkeypatch.setattr(demo, "installed_digest", lambda _: "digest")
    calls = []

    def ask(prompt, model, think):
        calls.append((prompt, model, think))
        return REPLY

    prompt = prompt_of(sr.K2)
    first = demo.live_opinion(folder, prompt, "m", ask)
    second = demo.live_opinion(folder, prompt, "m", ask)
    assert first == second and first["decision"] == sr.SUSPICIOUS
    assert calls == [(prompt, "m", False)]
    assert demo.cached_opinion(folder, prompt, "m") == first


@pytest.mark.parametrize(
    ("decision", "level", "agrees"),
    [
        (sr.SUSPICIOUS, sr.STRONG, True),
        (sr.SUSPICIOUS, sr.UNCERTAIN, True),
        (sr.NOT_SUSPICIOUS, sr.NOT_FLAGGED, True),
        (sr.NOT_SUSPICIOUS, sr.STRONG, False),
        (sr.SUSPICIOUS, sr.NOT_FLAGGED, False),
    ],
)
def test_agreement_follows_the_level_shown(decision, level, agrees):
    text = demo.agreement_text(decision, level)
    assert text.endswith("disagrees with the ML model.") != agrees


def test_no_agreement_without_a_decision_or_a_level():
    assert demo.agreement_text(None, sr.STRONG) is None
    assert demo.agreement_text(sr.SUSPICIOUS, None) is None


def test_a_new_decision_replaces_the_old(folder):
    assert demo.load_decisions(folder).empty
    demo.save_decision(folder, "Case 001", "Dismiss", "first")
    demo.save_decision(folder, "Case 001", "Escalate", "second")
    demo.save_decision(folder, "Case 002", "Dismiss", "")
    saved = demo.load_decisions(folder).set_index("case_name")
    assert len(saved) == 2
    assert saved.loc["Case 001", "decision"] == "Escalate"
    assert saved.loc["Case 001", "note"] == "second"


def results_frame(rows):
    return pd.DataFrame(rows, columns=["model", "account", "condition", "decision"])


def test_the_dose_table_counts_only_accounts_right_alone():
    reading = pd.DataFrame({"label": [1, 0, 1]}, index=["a", "b", "c"])
    bad, good = sr.NOT_SUSPICIOUS, sr.SUSPICIOUS
    rows = [
        # a is right alone and turns wrong at full dose.
        ("m", "a", sr.K1, good),
        ("m", "a", sr.K2, good),
        ("m", "a", sr.K3_D100, bad),
        # b is right alone and stays right.
        ("m", "b", sr.K1, bad),
        ("m", "b", sr.K2, bad),
        ("m", "b", sr.K3_D100, bad),
        # c is wrong alone, so it leaves the comparison.
        ("m", "c", sr.K1, bad),
        ("m", "c", sr.K2, good),
        ("m", "c", sr.K3_D100, bad),
    ]
    table = demo.dose_table(results_frame(rows), reading).set_index("condition")
    assert table.loc[sr.K2, "accounts"] == 2
    assert table.loc[sr.K2, "turned_wrong"] == 0
    assert table.loc[sr.K3_D100, "accounts"] == 2
    assert table.loc[sr.K3_D100, "turned_wrong"] == 1
    assert table.loc[sr.K3_D100, "share"] == 0.5
    assert np.isnan(table.loc[sr.K3_D33, "share"])


def test_an_invalid_answer_leaves_the_comparison():
    reading = pd.DataFrame({"label": [1, 1]}, index=["a", "b"])
    rows = [
        ("m", "a", sr.K1, sr.SUSPICIOUS),
        ("m", "a", sr.K2, None),
        ("m", "b", sr.K1, sr.SUSPICIOUS),
        ("m", "b", sr.K2, sr.NOT_SUSPICIOUS),
    ]
    table = demo.dose_table(results_frame(rows), reading).set_index("condition")
    assert table.loc[sr.K2, "accounts"] == 1
    assert table.loc[sr.K2, "share"] == 1.0


def test_models_are_counted_apart():
    reading = pd.DataFrame({"label": [1]}, index=["a"])
    rows = [
        ("m", "a", sr.K1, sr.SUSPICIOUS),
        ("m", "a", sr.K2, sr.NOT_SUSPICIOUS),
        ("n", "a", sr.K1, sr.NOT_SUSPICIOUS),
        ("n", "a", sr.K2, sr.NOT_SUSPICIOUS),
    ]
    table = demo.dose_table(results_frame(rows), reading)
    k2 = table[table["condition"] == sr.K2].set_index("model")
    assert k2.loc["m", "turned_wrong"] == 1
    assert k2.loc["n", "accounts"] == 0


def run_app():
    return AppTest.from_file(APP, default_timeout=30).run()


def metric_of(app, label):
    return next(m.value for m in app.metric if m.label == label)


def test_the_review_page_opens_without_labels(folder):
    app = run_app()
    assert not app.exception
    assert [t.value for t in app.title] == ["Case review"]
    assert metric_of(app, "Counterparties") == "16"
    assert metric_of(app, "Total sent (USD)") == "10,445,804"
    shown = " ".join(m.value for m in app.markdown)
    assert "Payment formats" in shown and "ACH 25" in shown
    assert "Counterparty:" in shown
    assert "1:A" not in shown and "label" not in shown.lower()


def test_both_models_stand_side_by_side(folder):
    prompt = prompt_of(sr.STRONG).replace(sr.STRONG, sr.K2)
    store_answer(folder, prompt)
    app = run_app()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "Gemma 4" in shown and "Qwen 3.5 9B" in shown
    assert metric_of(app, "Decision") == "suspicious"
    assert metric_of(app, "Its stated confidence") == "0.70"
    assert "relied on" in shown
    # One answer so far: no comparison, and the other model offers to be asked.
    assert not any("two models" in i.value for i in app.info)
    assert [b.label for b in app.button if b.label == "Ask this model"] == [
        "Ask this model"
    ]
    store_answer(folder, prompt, model="qwen3.5:9b-q4_K_M")
    app = run_app()
    assert any("same decision" in i.value for i in app.info)


def test_the_level_is_shown_as_a_band(folder):
    app = run_app()
    bands = [e.value for e in app.error] + [s.value for s in app.success]
    assert any(sr.STRONG in band or sr.NOT_FLAGGED in band for band in bands)


def test_the_analyst_decision_is_saved_from_the_page(folder):
    app = run_app()
    app.text_input(key="analyst_note").set_value("looks odd")
    app.radio(key="analyst_decision").set_value("Escalate")
    app.button(key="save_decision").click().run()
    assert not app.exception
    saved = demo.load_decisions(folder)
    assert saved["decision"].tolist() == ["Escalate"]
    assert saved["note"].tolist() == ["looks odd"]


def test_the_experiment_page_opens_with_and_without_results(folder):
    app = run_app()
    app.radio(key="page").set_value("Experiment").run()
    assert not app.exception
    assert any("No finished" in i.value for i in app.info)
    results = pd.DataFrame(
        {
            "model": ["m", "m"],
            "account": ["1:A", "1:A"],
            "condition": [sr.K1, sr.K3_D100],
            "decision": [sr.SUSPICIOUS, sr.NOT_SUSPICIOUS],
        }
    )
    results.to_parquet(folder / demo.RESULTS)
    app = run_app()
    app.radio(key="page").set_value("Experiment").run()
    assert not app.exception
    assert any("partial" in w.value for w in app.warning)


def test_without_pipeline_files_the_page_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv(demo.DIRECTORY_VARIABLE, str(tmp_path))
    app = run_app()
    assert any("run_pipeline" in e.value for e in app.error)


def test_the_research_view_shows_every_condition_and_warns(folder):
    app = run_app()
    app.sidebar.toggle[0].set_value(True).run()
    shown = app.sidebar.selectbox[0]
    assert shown.label == "What the reader is shown"
    assert len(shown.options) == len(sr.CONDITIONS)
    shown.set_value("No ML output (the reader alone)").run()
    assert not app.exception
    assert any("Hidden in this condition" in m.value for m in app.markdown)
    app.sidebar.selectbox[0].set_value("Corrupted ML output, 100%").run()
    assert any("deliberately corrupted" in w.value for w in app.warning)
