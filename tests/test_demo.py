"""Tests for the demo's logic and, through Streamlit's test runner, for its page."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from skeptical_second_reader import demo, fresh
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


def run_app(page="Case review"):
    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state["page"] = page
    return app.run()


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
    app.sidebar.selectbox[0].set_value("Reversed ML output, 100%").run()
    assert any("deliberately reversed" in w.value for w in app.warning)


# The reading test page.

FACTS_BLOCK = (
    "FACTS COMPUTED FROM ALL TRANSACTIONS OF THE WINDOW, INCLUDING THOSE OF THE "
    "COUNTERPARTIES\n- A is paid by 1 account and pays 1 account.\n\n"
    "HOW THESE FACTS COMPARE WITH THE OTHER ACCOUNTS OF THE BANK\n"
    "- different accounts that pay A: 1; higher than 45% of accounts."
)
GUIDED_ANSWER = (
    '{"decision": "not_suspicious", "confidence": 0.8, "score": 20, '
    '"evidence": [], "reason": "no sign"}'
)
MODEL_NAMES = ["gemma4:latest", "qwen3.5:9b-q4_K_M"]
READING_GROUPS = {
    "p:1": "pattern CYCLE",
    "p:2": "positive, outside the file",
    "c:1": "clean, matched activity",
    "c:2": "clean, random",
}


def evidence_of():
    return prompt_of(sr.K1).split("\n\n")[1]


def guided_prompt():
    shots = sr.shot_block([("4 transactions; paid by 1 accounts.", True, "one sign")])
    return sr.build_prompt(
        evidence_of(),
        None,
        facts=FACTS_BLOCK,
        guide="GUIDE: WHAT SEPARATES LAUNDERING ACCOUNTS IN THIS BANK\n- a sign",
        shots=shots,
        score=True,
    )


def reading_rows(condition):
    rows = []
    for account, group in READING_GROUPS.items():
        base = (
            sr.build_prompt(evidence_of(), None)
            if condition == pl.READ_ALONE
            else guided_prompt()
        )
        rows.append(
            {
                "account": account,
                "group": group,
                "condition": condition,
                # A distinct prompt per account, so each has its own cached answer.
                "prompt": base + f"\n\nCase {account}",
                "shown_level": None,
                "shown_rate": np.nan,
                "flipped": None,
            }
        )
    return rows


def reading_results(decisions):
    """decisions: {(account, condition): decision}, the same for both models."""
    rows = []
    for model in MODEL_NAMES:
        for (account, condition), decision in decisions.items():
            rows.append(
                {
                    "model": model,
                    "account": account,
                    "group": READING_GROUPS[account],
                    "condition": condition,
                    "decision": decision,
                    "confidence": 0.8,
                    "score": 20.0 if condition == pl.READ_SHOTS else np.nan,
                    "reason": "r",
                    "seconds": 1.0,
                }
            )
    return pd.DataFrame(rows)


S, N = sr.SUSPICIOUS, sr.NOT_SUSPICIOUS
DECISIONS = {
    ("p:1", pl.READ_ALONE): S,
    ("p:1", pl.READ_SHOTS): S,
    ("p:2", pl.READ_ALONE): S,
    ("p:2", pl.READ_SHOTS): N,
    ("c:1", pl.READ_ALONE): S,
    ("c:1", pl.READ_SHOTS): N,
    ("c:2", pl.READ_ALONE): N,
    ("c:2", pl.READ_SHOTS): N,
}


@pytest.fixture
def reading_folder(tmp_path, monkeypatch, folder):
    reading = tmp_path / "reading"
    reading.mkdir()
    alone, guided = reading_rows(pl.READ_ALONE), reading_rows(pl.READ_SHOTS)
    pd.DataFrame(alone).to_parquet(reading / "prompts_test.parquet")
    pd.DataFrame(guided).to_parquet(reading / "prompts_test_guided.parquet")
    results = reading_results(DECISIONS)
    is_alone = results["condition"] == pl.READ_ALONE
    results[is_alone].to_parquet(reading / "results_test.parquet")
    results[~is_alone].to_parquet(reading / "results_test_guided.parquet")
    cache = pl.AnswerCache(reading / demo.ANSWERS)
    reply = {**REPLY, "content": GUIDED_ANSWER}
    for model in MODEL_NAMES:
        digest = f"version of {model}"
        for row in alone + guided:
            cache.put(pl.answer_key(row["prompt"], digest, False), model, digest, reply)
    cache.close()
    accounts = list(READING_GROUPS)
    pd.DataFrame(
        {"score": [0.9, 0.2, 0.6, 0.1], "flag": [1, 0, 1, 0], "label": [1, 1, 0, 0]},
        index=accounts,
    ).to_parquet(tmp_path / "transaction_reading.parquet")
    pd.DataFrame(
        {
            "scores evidence and facts": [0.8, 0.3, 0.4, 0.1],
            "flags evidence and facts": [1, 0, 0, 0],
        },
        index=accounts,
    ).to_parquet(tmp_path / "ceiling_facts_scores.parquet")
    monkeypatch.setenv(demo.READING_VARIABLE, str(reading))
    monkeypatch.setenv(demo.SCORES_VARIABLE, str(tmp_path))
    return reading


def test_the_reading_sets_list_the_test_sample_and_fresh_runs(reading_folder):
    assert demo.reading_sets(reading_folder) == {demo.TEST_SET: (None, None)}
    run = reading_folder / "fresh_seed101_small"
    run.mkdir()
    pd.DataFrame(reading_rows(pl.READ_ALONE)).to_parquet(run / "prompts.parquet")
    assert list(demo.reading_sets(reading_folder)) == [demo.TEST_SET]
    reading_results(DECISIONS).to_parquet(run / demo.STORED_RESULTS)
    reading_results(DECISIONS).to_parquet(run / demo.LIVE_RESULTS)
    sets = demo.reading_sets(reading_folder)
    assert sets["Fresh sample, seed 101 (small)"] == (run, demo.STORED_RESULTS)
    assert sets["New accounts, seed 101 (small)"] == (run, demo.LIVE_RESULTS)


def test_the_test_sample_joins_alone_and_guided(reading_folder):
    prompts, results = demo.load_reading_set(reading_folder, None)
    assert set(prompts["condition"]) == {pl.READ_ALONE, pl.READ_SHOTS}
    assert len(prompts) == 2 * len(READING_GROUPS)
    assert len(results) == 2 * len(DECISIONS)


def test_reading_cases_hide_the_strata_in_a_fixed_order(reading_folder):
    prompts, _ = demo.load_reading_set(reading_folder, None)
    cases = demo.reading_cases(prompts)
    assert cases["case"].tolist() == [f"Reading {i:03d}" for i in range(1, 5)]
    assert set(cases["account"]) == set(READING_GROUPS)
    again = demo.reading_cases(prompts)
    assert cases.equals(again)


def test_a_guided_prompt_splits_into_its_sections():
    sections = demo.guided_sections(guided_prompt())
    assert sections["guide"].startswith("GUIDE:")
    assert len(sections["examples"]) == 1
    assert sections["examples"][0].startswith("Example 1: involved")
    assert sections["evidence"] == evidence_of()
    assert sections["facts"] == ["- A is paid by 1 account and pays 1 account."]
    assert sections["comparison"][0].startswith("- different accounts that pay A")


def test_a_prompt_without_the_guide_is_refused():
    with pytest.raises(ValueError):
        demo.guided_sections(sr.build_prompt(evidence_of(), None))


def test_demo_cases_follow_their_rules():
    groups = pd.Series(READING_GROUPS)
    chosen = demo.demo_cases(reading_results(DECISIONS), groups)
    assert chosen["c:1"] == "a false alarm the guide removes"
    assert chosen["p:1"] == "a positive caught with the guide"
    assert chosen["p:2"] == "a positive outside the pattern file"
    assert "c:2" not in chosen.index
    assert demo.demo_cases(pd.DataFrame(), groups).empty


def test_a_demo_case_needs_both_models_to_agree():
    results = reading_results(DECISIONS)
    gemma = (results["model"] == MODEL_NAMES[0]) & (results["account"] == "c:1")
    results.loc[gemma & (results["condition"] == pl.READ_SHOTS), "decision"] = S
    chosen = demo.demo_cases(results, pd.Series(READING_GROUPS))
    assert "c:1" not in chosen.index


def test_missing_stored_answers_are_counted(reading_folder):
    prompts, _ = demo.load_reading_set(reading_folder, None)
    assert demo.cache_gaps(reading_folder, prompts, READING_GROUPS, MODEL_NAMES) == 0
    assert demo.cache_gaps(reading_folder, prompts, ["p:1"], ["other:model"]) == 2


def test_a_live_check_asks_again_and_leaves_the_stored_answers(
    reading_folder, monkeypatch
):
    model = MODEL_NAMES[0]
    monkeypatch.setattr(demo, "installed_digest", lambda m: f"version of {m}")
    calls = []

    def ask(text, name, think):
        calls.append((name, think))
        return {**REPLY, "content": GUIDED_ANSWER}

    stored = pl.AnswerCache(reading_folder / demo.ANSWERS)
    count = len(stored)
    stored.close()
    demo.live_check(reading_folder, guided_prompt(), model, ask=ask)
    live = demo.live_check(reading_folder, guided_prompt(), model, ask=ask)
    assert calls == [(model, False), (model, False)]
    assert live["decision"] == N and live["score"] == 20 and live["same_version"]
    stored = pl.AnswerCache(reading_folder / demo.ANSWERS)
    assert len(stored) == count
    stored.close()
    kept = pl.AnswerCache(reading_folder / demo.LIVE_ANSWERS)
    assert len(kept) == 1
    kept.close()


def test_a_live_check_says_when_the_model_version_changed(reading_folder, monkeypatch):
    monkeypatch.setattr(demo, "installed_digest", lambda m: "a newer version")
    live = demo.live_check(
        reading_folder,
        guided_prompt(),
        MODEL_NAMES[0],
        ask=lambda *a, **k: {**REPLY, "content": GUIDED_ANSWER},
    )
    assert not live["same_version"]


def test_live_and_stored_answers_are_compared():
    stored = {"decision": S, "confidence": 0.7, "score": 60}
    assert "repeats" in demo.same_answer_text(stored, dict(stored))
    changed = demo.same_answer_text(stored, {**stored, "score": 70})
    assert "differs" in changed and "score" in changed and "decision" not in changed
    assert "No stored answer" in demo.same_answer_text(None, stored)


def test_the_guide_change_is_described():
    assert demo.guide_change_text(S, S) == "The guide did not change the decision."
    assert "from suspicious to not suspicious" in demo.guide_change_text(S, N)
    assert demo.guide_change_text(None, S) is None


def test_the_ml_view_has_no_label(reading_folder):
    view = demo.ml_view(["c:1", "p:1"], demo.scores_directory())
    assert list(view.columns) == ["score", "flag"]
    assert view.index.tolist() == ["c:1", "p:1"]


def test_a_live_run_is_estimated_from_measured_call_times():
    gemma = demo.live_minutes(20, [MODEL_NAMES[0]])
    assert gemma == pytest.approx((2 * 20 * 5 + 15) / 60)
    both = demo.live_minutes(5, MODEL_NAMES)
    assert both == pytest.approx((2 * 5 * 5 + 15 + 2 * 5 * 21 + 15) / 60)


def test_a_fresh_run_starts_in_the_background(tmp_path):
    commands = []

    def popen(command, **kwargs):
        commands.append((command, kwargs))

    folder = demo.start_fresh_run(
        tmp_path, 102, "full", [MODEL_NAMES[0]], accounts=20, live=True, popen=popen
    )
    command, kwargs = commands[0]
    assert folder == tmp_path / "fresh_seed102_full"
    assert command[1].endswith("run_fresh.py")
    assert command[2:] == [
        "--seed",
        "102",
        "--size",
        "full",
        "--models",
        MODEL_NAMES[0],
        "--live",
        "--accounts",
        "20",
    ]
    assert kwargs["cwd"] == demo.PROJECT_ROOT
    kwargs["stdout"].close()
    assert (folder / demo.LIVE_LOG).exists()
    with pytest.raises(ValueError):
        demo.start_fresh_run(tmp_path, 12, "mini", ["m"], popen=popen)


def test_the_run_status_is_read_from_the_log(tmp_path):
    log = tmp_path / demo.FRESH_LOG
    assert demo.fresh_run_status(tmp_path) == ("not started", "")
    log.write_text("answers: m\n  25/60  4.7s\n", encoding="utf-8")
    assert demo.fresh_run_status(tmp_path) == ("running", "  25/60  4.7s")
    log.write_text("Traceback (most recent call last):\nX\n", encoding="utf-8")
    assert demo.fresh_run_status(tmp_path)[0] == "failed"
    log.write_text("table\nfinished\n", encoding="utf-8")
    assert demo.fresh_run_status(tmp_path) == ("finished", "finished")


def test_a_result_row_becomes_an_opinion():
    results = reading_results(DECISIONS)
    opinion = demo.result_opinion(results, "c:1", pl.READ_SHOTS, MODEL_NAMES[0])
    assert opinion["decision"] == N and opinion["score"] == 20.0
    alone = demo.result_opinion(results, "c:1", pl.READ_ALONE, MODEL_NAMES[0])
    assert alone["score"] is None
    assert demo.result_opinion(results, "x", pl.READ_ALONE, MODEL_NAMES[0]) is None


def test_the_guided_page_opens_without_labels(reading_folder):
    app = run_app("Guided reading")
    assert not app.exception
    assert [t.value for t in app.title] == ["Guided reading"]
    shown = " ".join(m.value for m in app.markdown)
    assert "Reading alone" in shown and "With the guide and examples" in shown
    assert "Gemma 4" in shown and "Qwen 3.5 9B" in shown
    for account, group in READING_GROUPS.items():
        assert account not in shown and group not in shown
    assert all("label" not in frame.value.columns for frame in app.dataframe)
    assert any("Every answer" in s.value for s in app.sidebar.success)
    assert metric_of(app, "Score 0-100") == "20"


def test_the_research_view_shows_the_stratum_and_the_readers(reading_folder):
    app = run_app("Guided reading")
    app.toggle(key="reading_research").set_value(True).run()
    assert not app.exception
    assert any("Stratum:" in i.value for i in app.info)
    readers = app.dataframe[-1].value
    assert "XGBoost transaction model" in readers["reader"].tolist()
    assert any(name.endswith(pl.READ_SHOTS) for name in readers["reader"])


def test_a_positive_counts_as_caught_only_if_both_models_catch_it():
    results = reading_results(DECISIONS)
    qwen = (results["model"] == MODEL_NAMES[1]) & (results["account"] == "p:1")
    results.loc[qwen & (results["condition"] == pl.READ_SHOTS), "decision"] = N
    chosen = demo.demo_cases(results, pd.Series(READING_GROUPS))
    assert chosen.get("p:1") != "a positive caught with the guide"


# How it is checked.


def test_the_tests_of_the_repository_are_counted(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "def test_one():\n    pass\n\n\ndef helper():\n    pass\n\n\n"
        "def test_two():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "conftest.py").write_text("def test_no(): pass\n")
    assert demo.test_count(tmp_path) == 2
    assert demo.test_count() > 200


def test_the_last_mutation_run_is_read_if_saved(tmp_path):
    assert demo.mutation_summary(tmp_path) is None
    (tmp_path / demo.MUTATION_REPORT).write_text(
        '{"date": "2026-11-18 10:00", "total": 3, "caught": 3, "missed": []}',
        encoding="utf-8",
    )
    assert demo.mutation_summary(tmp_path)["caught"] == 3


def test_the_assurance_table_names_what_each_step_guards_against():
    table = demo.assurance_table()
    assert list(table.columns) == ["What we did", "Guards against", "Where"]
    assert table.notna().all().all() and len(table) >= 5


def test_the_checks_page_opens_with_and_without_a_mutation_run(
    reading_folder, monkeypatch
):
    monkeypatch.setattr(demo, "collected_tests", lambda: 279)
    app = run_app("How it is checked")
    assert not app.exception
    assert metric_of(app, "Tests") == "279"
    assert [t.value for t in app.title] == ["How it is checked"]
    assert metric_of(app, "Planted mistakes caught") == "not run yet"
    scores = demo.scores_directory()
    (scores / demo.MUTATION_REPORT).write_text(
        '{"date": "d", "total": 5, "caught": 4, "missed": ["x"]}', encoding="utf-8"
    )
    app = run_app("How it is checked")
    assert metric_of(app, "Planted mistakes caught") == "4 of 5"
    assert any("Missed: x" in w.value for w in app.warning)


# The live run page.


def test_a_live_run_records_its_plan_and_starts_a_new_answers_file(tmp_path):
    folder = tmp_path / "fresh_seed102_full"
    folder.mkdir()
    (folder / fresh.LIVE_ANSWERS).write_text("old run\n", encoding="utf-8")

    def popen(command, **kwargs):
        kwargs["stdout"].close()

    demo.start_fresh_run(
        tmp_path, 102, "full", MODEL_NAMES, accounts=3, live=True, popen=popen
    )
    assert not (folder / fresh.LIVE_ANSWERS).exists()
    plan = demo.live_plan(folder)
    assert plan == {"models": MODEL_NAMES, "accounts": 3}
    assert demo.expected_answers(plan) == 12
    assert demo.expected_answers({"models": ["m"], "accounts": None}) is None
    assert demo.expected_answers(None) is None


def test_the_log_tail_keeps_the_last_lines(tmp_path):
    assert demo.log_tail(tmp_path) == ""
    lines = "\n".join(f"line {i}" for i in range(20))
    (tmp_path / demo.LIVE_LOG).write_text(lines + "\n\n", encoding="utf-8")
    assert demo.log_tail(tmp_path, lines=3) == "line 17\nline 18\nline 19"


def test_reader_names_are_written_out():
    models = {MODEL_NAMES[0]: "Gemma 4", MODEL_NAMES[1]: "Qwen 3.5 9B"}
    assert demo.reader_name("gemma4 " + pl.READ_ALONE, models) == "Gemma 4, alone"
    shots = "qwen3.5 " + pl.READ_SHOTS
    assert demo.reader_name(shots, models) == "Qwen 3.5 9B, with the guide"
    xgb = "XGBoost transaction model"
    assert demo.reader_name(xgb, models) == xgb


def test_live_answers_are_shown_with_case_names_and_no_accounts():
    answers = pd.DataFrame(
        {
            "time": ["10:00:01", "10:00:05"],
            "model": [MODEL_NAMES[0]] * 2,
            "account": ["p:1", "c:1"],
            "condition": [pl.READ_ALONE, pl.READ_SHOTS],
            "decision": [S, None],
            "confidence": [0.8, None],
            "seconds": [4.04, 5.0],
            "source": ["live", "live"],
        }
    )
    cases = pd.DataFrame(
        {"case": ["Reading 001", "Reading 002"], "account": ["p:1", "c:1"]}
    )
    table = demo.live_answer_table(answers, cases, {MODEL_NAMES[0]: "Gemma 4"})
    assert table["case"].tolist() == ["Reading 002", "Reading 001"]  # newest first
    assert table["decision"].tolist() == ["no usable answer", "suspicious"]
    assert table["reading"].tolist() == ["with the guide", "alone"]
    assert set(table["model"]) == {"Gemma 4"}
    assert "account" not in table.columns and "p:1" not in table.to_string()


def test_the_outputs_of_a_live_run_are_listed(tmp_path):
    folder = tmp_path / "fresh_seed102_full"
    folder.mkdir()
    (folder / demo.LIVE_LOG).write_text("finished\n", encoding="utf-8")
    table = demo.live_outputs(folder)
    assert list(table.columns) == ["file", "holds", "last written"]
    written = dict(zip(table["file"], table["last written"], strict=True))
    log = next(name for name in written if name.endswith(demo.LIVE_LOG))
    assert written[log] != "not yet"
    assert any(name.endswith("live_answers.sqlite") for name in written)
    assert sum(value == "not yet" for value in written.values()) == 4


def test_full_reversal_is_counted_per_group_where_the_output_is_wrong():
    reading = pd.DataFrame({"label": [1, 1, 0, 1]}, index=["a", "b", "c", "d"])
    rows = []

    def add(account, group, condition, decision, level=None):
        rows.append(
            {
                "model": "m",
                "account": account,
                "group": group,
                "condition": condition,
                "decision": decision,
                "shown_level": level,
            }
        )

    right = "strong, model right"
    clean = "not flagged, clean"
    missed = "missed, outside the file"
    # a and b are right alone; the reversed output says not flagged, which is wrong.
    add("a", right, sr.K1, S)
    add("a", right, sr.K3_D100, N, sr.NOT_FLAGGED)
    add("b", right, sr.K1, S)
    add("b", right, sr.K3_D100, S, sr.NOT_FLAGGED)
    # c is wrong alone, so it is not counted.
    add("c", clean, sr.K1, S)
    add("c", clean, sr.K3_D100, S, sr.STRONG)
    # d: the ML model missed it, so the reversed output (an alert) is right.
    add("d", missed, sr.K1, S)
    add("d", missed, sr.K3_D100, S, sr.STRONG)
    table = demo.reversal_table(pd.DataFrame(rows), reading)
    table = table.set_index("ML outcome group")
    assert table.loc[right, "m"] == "1 of 2"
    assert table.loc[right, "shown after reversal"] == "not flagged"
    assert table.loc[clean, "m"] == "0 of 0"
    assert table.loc[clean, "shown after reversal"] == "alert"
    assert table.loc[missed, "m"] == "output right: not counted"
    assert table.loc["Total", "m"] == "1 of 2"
    assert "uncertain, model right" not in table.index


def test_the_collected_tests_are_read_from_pytest():
    class Done:
        stdout = "a\nb\n279 tests collected in 4.9s\n"

    class Nothing:
        stdout = "error"

    def broken(*args, **kwargs):
        raise OSError("no python")

    assert demo.collected_tests(run=lambda *a, **k: Done()) == 279
    assert demo.collected_tests(run=lambda *a, **k: Nothing()) is None
    assert demo.collected_tests(run=broken) is None


def test_the_experiment_page_shows_the_groups_when_the_run_is_complete(folder):
    rows = []
    cases = [
        ("1:A", "strong, model right", S, sr.NOT_FLAGGED),
        ("2:B", "not flagged, clean", N, sr.STRONG),
    ]
    for model in MODEL_NAMES:
        for account, group, alone, reversed_level in cases:
            for condition in sr.CONDITIONS:
                full = condition == sr.K3_D100
                rows.append(
                    {
                        "model": model,
                        "account": account,
                        "group": group,
                        "condition": condition,
                        "shown_level": reversed_level if full else None,
                        # Right alone, and follows the reversed output.
                        "decision": (S if alone == N else N) if full else alone,
                    }
                )
    pd.DataFrame(rows).to_parquet(folder / demo.RESULTS)
    app = run_app("Experiment")
    assert not app.exception
    assert any("Every answer is stored" in s.value for s in app.success)
    groups = next(
        f.value for f in app.dataframe if "ML outcome group" in f.value.columns
    )
    total = groups.set_index("ML outcome group").loc["Total"]
    assert total["Gemma 4"] == "2 of 2" and total["Qwen 3.5 9B"] == "2 of 2"


def test_the_run_page_without_a_pool_says_how_to_build_it(reading_folder):
    app = run_app("Run new accounts")
    assert not app.exception
    assert any("--prepare" in i.value for i in app.info)


def test_the_run_page_shows_a_finished_run_without_labels(reading_folder):
    pool = reading_folder / "fresh_seed102_full"
    pool.mkdir()
    prompts = reading_rows(pl.READ_ALONE) + reading_rows(pl.READ_SHOTS)
    pd.DataFrame(prompts).to_parquet(pool / "prompts.parquet")
    for account in ["p:1", "c:1"]:
        for condition in (pl.READ_ALONE, pl.READ_SHOTS):
            fresh.append_live_answer(
                pool / fresh.LIVE_ANSWERS,
                {
                    "time": "10:00:00",
                    "model": MODEL_NAMES[0],
                    "account": account,
                    "condition": condition,
                    "decision": S,
                    "confidence": 0.8,
                    "seconds": 4.0,
                    "source": "live",
                },
            )
    log = "step 1/4  accounts\nfinished\n"
    (pool / demo.LIVE_LOG).write_text(log, encoding="utf-8")
    pd.DataFrame(
        {"reader": ["gemma4 K1"], "accounts": [2], "AUC-ROC": [1.0], "recall": [1.0]}
    ).to_csv(pool / demo.LIVE_SUMMARY, index=False)
    app = run_app("Run new accounts")
    assert not app.exception
    assert any("finished" in s.value for s in app.success)
    frames = [f.value for f in app.dataframe]
    assert any("decision" in f.columns and len(f) == 4 for f in frames)
    assert any("XGBoost score" in f.columns for f in frames)
    assert any("reader" in f.columns for f in frames)
    assert any("holds" in f.columns for f in frames)
    shown = " ".join(f.to_string() for f in frames)
    shown += " ".join(m.value for m in app.markdown)
    for account, group in READING_GROUPS.items():
        assert account not in shown and group not in shown


def test_ml_overview_pools_precision_and_recall_and_averages_auc_pr_over_folds():
    reading = pd.DataFrame(
        {
            "score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            "flag": [1, 0, 1, 1, 0, 0],
            "label": [1, 0, 1, 0, 1, 0],
            "fold": [0, 0, 1, 1, 1, 1],
        }
    )
    row = demo.ml_overview(reading).iloc[0]
    assert row["accounts"] == 6 and row["positives"] == 3 and row["flagged"] == 3
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(2 / 3)
    # Each fold ranks its positives first, so both folds reach 1.
    assert row["AUC-PR (mean of folds)"] == pytest.approx(1.0)


def test_reversal_by_direction_splits_the_total_by_the_wrong_output_shown():
    rows = []
    # Positive shown 'not flagged' and follows; clean shown 'alert' and resists.
    for account, alone, shown, answer in [
        ("1:A", S, sr.NOT_FLAGGED, N),
        ("2:B", N, sr.STRONG, N),
    ]:
        rows.append(
            {
                "model": "m",
                "account": account,
                "condition": sr.K1,
                "decision": alone,
                "shown_level": None,
            }
        )
        rows.append(
            {
                "model": "m",
                "account": account,
                "condition": sr.K3_D100,
                "decision": answer,
                "shown_level": shown,
            }
        )
    reading = pd.DataFrame({"label": [1, 0]}, index=["1:A", "2:B"])
    table = demo.reversal_by_direction(pd.DataFrame(rows), reading)
    table = table.set_index("wrong output shown")["m"]
    assert table.iloc[0] == "1 of 1"
    assert table.iloc[1] == "0 of 1"
    assert table.loc["Total"] == "1 of 2"


def test_reader_metrics_adds_precision_and_the_number_called_suspicious():
    strata = pd.Series(
        ["pattern CYCLE", "pattern CYCLE", fresh.CLEAN[0], fresh.CLEAN[1]],
        index=["a", "b", "c", "d"],
    )
    readers = {
        "r": (
            pd.Series([0.9, 0.2, 0.8, 0.1], index=strata.index),
            pd.Series([True, False, True, False], index=strata.index),
        )
    }
    row = demo.reader_metrics(strata, readers).iloc[0]
    assert row["called suspicious"] == 2
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["false alarm rate"] == pytest.approx(0.5)


def test_suspicious_share_is_the_share_called_suspicious_per_stratum():
    results = pd.DataFrame(
        {
            "model": ["m"] * 3,
            "condition": [pl.READ_ALONE] * 3,
            "account": ["a", "b", "c"],
            "decision": [S, N, S],
        }
    )
    strata = pd.Series(["x", "x", "y"], index=["a", "b", "c"])
    table = demo.suspicious_share(results, strata)
    assert table.loc["x", ("m", pl.READ_ALONE)] == pytest.approx(0.5)
    assert table.loc["y", ("m", pl.READ_ALONE)] == pytest.approx(1.0)


def test_the_experiment_page_opens_without_scores_or_reading_files(
    folder, tmp_path, monkeypatch
):
    # The page caches its tables; another test may have filled the cache.
    st.cache_data.clear()
    empty = tmp_path / "nothing"
    empty.mkdir()
    monkeypatch.setenv(demo.SCORES_VARIABLE, str(empty))
    monkeypatch.setenv(demo.READING_VARIABLE, str(empty))
    rows = [
        {
            "model": m,
            "account": a,
            "condition": c,
            "decision": S,
            "group": "g",
            "shown_level": sr.NOT_FLAGGED,
        }
        for m in MODEL_NAMES
        for a in ["1:A", "2:B"]
        for c in sr.CONDITIONS
    ]
    pd.DataFrame(rows).to_parquet(folder / demo.RESULTS)
    app = run_app("Experiment")
    assert not app.exception
    assert any("ML scores are not found" in i.value for i in app.info)
    assert any("reading test are not found" in i.value for i in app.info)
