"""The demo: an analyst reviews flagged accounts with an LLM second reader.

Run from the project root:

    uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from skeptical_second_reader import demo, fresh
from skeptical_second_reader import pipeline as pl
from skeptical_second_reader import second_reader as sr

MODELS = {"gemma4:latest": "Gemma 4", "qwen3.5:9b-q4_K_M": "Qwen 3.5 9B"}
CONDITION_NAMES = {
    sr.K2: "True ML output",
    sr.K1: "No ML output (the reader alone)",
    sr.K3_D33: "Reversed ML output, 33%",
    sr.K3_D66: "Reversed ML output, 66%",
    sr.K3_D100: "Reversed ML output, 100%",
    sr.SCORE_ONLY: "Reversed ML output, 100%, no reasons",
}

st.set_page_config(page_title="Skeptical Second Reader", layout="wide")
directory = demo.data_directory()


@st.cache_data
def load_prompts(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


if not (directory / demo.PROMPTS).exists():
    st.error("No pipeline output found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

prompts = load_prompts(str(directory / demo.PROMPTS))
cases = demo.load_cases(directory)
PAGES = [
    "Guided reading",
    "Case review",
    "Run new accounts",
    "Experiment",
    "How it is checked",
]
page = st.sidebar.radio("Page", PAGES, key="page")
st.sidebar.caption(
    "Synthetic data (IBM AML HI-Small). A research prototype, not a compliance tool."
)


def show_level(level: str) -> None:
    """The level as a coloured band; the words carry it without the colour too."""
    band = {sr.STRONG: st.error, sr.UNCERTAIN: st.warning, sr.NOT_FLAGGED: st.success}
    band[level](f"**ML level: {level}**")


def show_summary(summary: dict) -> None:
    first = st.columns(4)
    first[0].metric("Transactions", summary["transactions"], border=True)
    first[1].metric("Sent", summary["sent"], border=True)
    first[2].metric("Received", summary["received"], border=True)
    first[3].metric("To itself", summary["to_itself"], border=True)
    second = st.columns(4)
    second[0].metric("Currency conversions", summary["conversions"], border=True)
    second[1].metric("Counterparties", summary["counterparties"], border=True)
    second[2].metric("Total sent (USD)", f"{summary['sent_usd']:,.0f}", border=True)
    second[3].metric(
        "Total received (USD)", f"{summary['received_usd']:,.0f}", border=True
    )
    st.write(f"**Payment formats:** {summary['formats']}")
    if summary["shown"] < summary["transactions"]:
        st.caption(
            f"The table shows the {summary['shown']} largest of "
            f"{summary['transactions']} transactions, chosen by amount, not by the "
            "ML model."
        )


def show_opinion(opinion: dict, level: str | None) -> None:
    decision = opinion["decision"]
    if decision is None:
        st.warning("The model gave no usable answer to this prompt.")
        return
    st.metric("Decision", decision.replace("_", " "), border=True)
    confidence = opinion["confidence"]
    st.metric(
        "Its stated confidence",
        "n/a" if confidence is None else f"{confidence:.2f}",
        border=True,
    )
    text = demo.agreement_text(decision, level)
    if text:
        (st.warning if "disagrees" in text else st.success)(text)
    if opinion["evidence"]:
        numbers = ", ".join(str(number) for number in opinion["evidence"])
        st.write(f"**Transactions it relied on:** {numbers}")
    st.write(opinion["reason"] or "")
    st.caption(f"Answered in {opinion['seconds']:.1f} s")


def review_page() -> None:
    st.title("Case review")
    levels = st.sidebar.multiselect(
        "ML level",
        sorted(cases["level"].unique()),
        default=sorted(cases["level"].unique()),
    )
    visible = cases[cases["level"].isin(levels)]
    if visible.empty:
        st.info("No cases at these levels.")
        return
    names = visible["level"].map(demo.LEVEL_MARKS) + " " + visible["case"]
    names = names + "  ·  " + visible["level"]
    chosen = st.selectbox("Case", names.tolist())
    case = visible[names == chosen].iloc[0]

    research = st.sidebar.toggle("Research view", value=False)
    condition = sr.K2
    if research:
        shown = st.sidebar.selectbox(
            "What the reader is shown", list(CONDITION_NAMES.values())
        )
        condition = {name: key for key, name in CONDITION_NAMES.items()}[shown]
        if condition not in (sr.K1, sr.K2):
            st.warning("This ML output is deliberately reversed for the experiment.")
    row = demo.case_prompt(prompts, case["account"], condition)
    parts = demo.split_prompt(row["prompt"])

    st.subheader("Evidence")
    show_summary(demo.parse_summary(parts["summary"]))
    transactions = parts["transactions"].rename(columns=demo.TABLE_LABELS)
    st.dataframe(
        transactions,
        hide_index=True,
        width="stretch",
        column_config={
            demo.TABLE_LABELS["amount_usd"]: st.column_config.NumberColumn(
                format="%,.2f"
            )
        },
    )
    with st.expander("What do these terms mean?"):
        for term, meaning in demo.GLOSSARY.items():
            st.markdown(f"**{term}:** {meaning}")

    st.subheader("ML model")
    if parts["ml_text"] is None:
        st.write("Hidden in this condition.")
    else:
        show_level(row["shown_level"])
        st.text(demo.ml_details(parts["ml_text"]))

    st.subheader("Second reader")
    level = None if parts["ml_text"] is None else row["shown_level"]
    opinions = {
        model: demo.cached_opinion(directory, row["prompt"], model) for model in MODELS
    }
    comparison = demo.models_comparison(
        [o["decision"] if o else None for o in opinions.values()]
    )
    if comparison:
        st.info(comparison)
    for column, (model, opinion) in zip(
        st.columns(len(MODELS)), opinions.items(), strict=True
    ):
        with column:
            st.markdown(f"#### {MODELS[model]}")
            if opinion is not None:
                show_opinion(opinion, level)
                continue
            st.write("No stored answer for this case yet.")
            if st.button("Ask this model", key=f"ask {model}"):
                try:
                    with st.spinner("The local model is reading the case..."):
                        show_opinion(
                            demo.live_opinion(directory, row["prompt"], model), level
                        )
                except OSError as error:
                    st.error(f"The local model could not be reached: {error}")

    st.subheader("Analyst decision")
    with st.form("decision"):
        decision = st.radio(
            "Your decision",
            ["Escalate", "Dismiss", "Needs more information"],
            horizontal=True,
            key="analyst_decision",
        )
        note = st.text_input("Note", key="analyst_note")
        if st.form_submit_button("Save decision", key="save_decision"):
            demo.save_decision(directory, case["case"], decision, note)
            st.success(f"Saved for {case['case']}.")
    saved = demo.load_decisions(directory)
    if not saved.empty:
        with st.expander(f"Saved decisions ({len(saved)})"):
            st.dataframe(saved, hide_index=True, width="stretch")


EXPERIMENT_INTRO = """
**Question.** When the evidence stays the same and only the ML output changes, does a
local LLM change its decision with the output?

**From the data to the answer**
1. **Data.** IBM AML HI-Small, synthetic: 5.08 M transactions between 515 k accounts,
   1-18 September 2022. An account is positive if it sent or received a laundering
   transaction (1.2%). Features use 1-10 September only, because after 10 September
   most transactions are laundering and a model could learn the calendar.
2. **Split.** Accounts, not transactions, go into 5 folds (stratified group k-fold).
   All accounts of one laundering pattern form one group, and the other accounts are
   grouped by owner, so no laundering attempt crosses folds. Each round: 1 test fold,
   1 validation fold (early stopping, threshold), 3 training folds. Every account is
   scored by a model that never saw it.
3. **ML model.** Transaction-level XGBoost; an account's score combines its
   transaction scores (noisy-or). Threshold: lowest score with validation precision
   of at least 0.7. Three strongest reasons per account from SHAP.
4. **Prompt.** The account's 30 largest transactions and a summary line;
   counterparties masked; no label, pattern name or identity. Depending on the
   condition, the ML output is added: true, or reversed with its reasons.
5. **Readers.** Gemma 4 and Qwen 3.5 9B, 4-bit, through Ollama on this laptop;
   temperature 0, context 4,096, thinking off. Answer: decision, confidence, reason,
   cited transactions. Every answer is cached under prompt + model version + settings.
6. **Measures.** Experiment 1: among accounts a model decides correctly alone, how
   many turn wrong when shown a wrong output. Experiment 2: AUC-ROC, precision, recall
   and false alarm rate, without any ML output.

**Constraints.** Synthetic data; small 4-bit models on one laptop (16 GB memory, 6 GB
GPU); English prompts. Some owners are still split across folds, so the claim is about
a bank's existing customers. The strongest signal is probably a mark of the
simulator. A research prototype, not a compliance tool.
"""


@st.cache_data
def ml_overview_table() -> pd.DataFrame | None:
    path = demo.scores_directory() / "transaction_reading.parquet"
    return demo.ml_overview(pd.read_parquet(path)) if path.exists() else None


@st.cache_data
def guided_tables() -> tuple | None:
    """Experiment 2 on the test sample and on the third sample (seed 102)."""
    folder = demo.reading_directory()
    if demo.TEST_SET not in demo.reading_sets(folder):
        return None
    _, test = demo.load_reading_set(folder, None)
    strata = test.drop_duplicates("account").set_index("account")["group"]
    test_table = demo.reader_metrics(strata, fresh.llm_readers(test))
    shares = demo.suspicious_share(test, strata)
    third_folder = folder / fresh.run_name(demo.POOL_SEED, demo.POOL_SIZE)
    third_table = None
    scores = demo.scores_directory()
    needed = [
        third_folder / demo.STORED_RESULTS,
        third_folder / "cases.parquet",
        scores / "transaction_reading.parquet",
        scores / "ceiling_facts_scores.parquet",
    ]
    if all(path.exists() for path in needed):
        third = pd.read_parquet(third_folder / demo.STORED_RESULTS)
        third_strata = pd.read_parquet(third_folder / "cases.parquet")["stratum"]
        readers = fresh.ml_readers(
            third_strata.index,
            pd.read_parquet(scores / "transaction_reading.parquet"),
            pd.read_parquet(scores / "ceiling_facts_scores.parquet"),
        ) | fresh.llm_readers(third)
        third_table = demo.reader_metrics(third_strata, readers)
    return test_table, shares, third_table


def readable(table: pd.DataFrame) -> pd.DataFrame:
    table = table.assign(reader=[demo.reader_name(r, MODELS) for r in table["reader"]])
    return table.round(3)


def experiment_page() -> None:
    st.title("Experiment")
    st.markdown(EXPERIMENT_INTRO)

    st.subheader("ML model: all accounts, test folds")
    overview = ml_overview_table()
    if overview is None:
        st.info("The stored ML scores are not found.")
    else:
        st.dataframe(overview.round(3), hide_index=True, width="stretch")
        st.caption(
            "Out-of-fold scores at the model's threshold. A random ranking reaches an "
            "AUC-PR of about the positive rate (0.012)."
        )

    st.header("Experiment 1: does the reader follow a reversed ML output?")
    st.write(
        "203 accounts, 29 from each of seven ML outcome groups, each read in six "
        "conditions: no ML output, the true output, the output reversed at 33%, 66% "
        "and 100%, and the reversed score without reasons. The evidence never changes."
    )
    st.markdown(
        "**How the output is reversed.** The calibrated rate shown to the reader is "
        "moved towards the far end of the opposite decision: new rate = rate + dose "
        "x (target - rate). Target: **0.01** if the model flagged the account, "
        "**0.99** if not. Dose: **0.33 · 0.66 · 1.0**. The level is recomputed from "
        "the new rate, and the direction of the reasons and the count of high "
        "transactions are reversed with it. The transactions never change."
    )
    progress = demo.progress_table(directory)
    expected = len(prompts)
    if progress.empty:
        st.info("No finished results yet.")
        return
    unfinished = (progress["answers"] < expected).any() or len(progress) < len(MODELS)
    if unfinished:
        for entry in progress.itertuples():
            st.progress(
                min(entry.answers / expected, 1.0),
                f"{entry.model}: {entry.answers} of {expected}",
            )
        st.warning("The run is not finished: numbers below are partial.")
    else:
        st.success(
            f"Every answer is stored: {expected:,} per model ({len(cases)} accounts "
            f"in {len(sr.CONDITIONS)} conditions)."
        )
    results = pd.read_parquet(directory / demo.RESULTS)
    reading = pd.read_parquet(directory / demo.READING, columns=["label"])

    if {"group", "shown_level"} <= set(results.columns):
        st.subheader("Full reversal, by the direction of the wrong output")
        direction = demo.reversal_by_direction(results, reading).rename(columns=MODELS)
        st.dataframe(direction, hide_index=True, width="stretch")
        st.caption(
            "Counted: the accounts a model decides correctly alone and whose reversed "
            "output is wrong; each cell says how many changed to the wrong answer."
        )
        st.subheader("Full reversal, per ML outcome group")
        groups = demo.reversal_table(results, reading).rename(columns=MODELS)
        st.dataframe(groups, hide_index=True, width="stretch")
        st.caption(
            "Counted: the accounts a model decides correctly alone and whose reversed "
            "output is wrong; each cell says how many changed to the wrong answer. "
            "Where the ML model is wrong the reversed output is right, so those groups "
            "are not counted. Notebook 11 gives the same counts with intervals."
        )

    st.subheader("Every dose, all groups pooled")
    table = demo.dose_table(results, reading)
    table["condition"] = table["condition"].map(CONDITION_NAMES)
    table["model"] = table["model"].map(MODELS).fillna(table["model"])
    st.dataframe(table, hide_index=True, width="stretch")
    st.bar_chart(
        table, x="condition", y="share", color="model", stack=False, horizontal=True
    )
    st.caption(
        "Pooled over groups: where the ML model is right a reversed output is wrong, "
        "where it is wrong a reversed output is right. Sample of equal groups, so the "
        "shares do not describe the population."
    )

    st.header("Is the evidence readable?")
    st.write(
        "Before judging the readers, a classifier was trained only on what the prompt "
        "shows. On the 203 accounts of Experiment 1 it reached AUC-ROC 0.57 (the LLMs "
        "0.49-0.54): that sample was drawn by ML errors and its evidence was weak. New "
        "samples were drawn by what an account is (15 of each of eight pattern types, "
        "15 positives outside the pattern file, 60 matched clean, 60 random clean; 255 "
        "accounts). On the development sample the classifier reached 0.86, so the "
        "evidence is readable (notebooks 12 and 13)."
    )

    st.header("Experiment 2: reading with a guide")
    st.write(
        "No ML output is shown. Each account is read alone and with a guide that puts "
        "each fact beside clean reference accounts, plus four solved examples. The "
        "version was chosen on a development sample by a rule fixed before the run; "
        "the test sample was read once (notebooks 15 and 16)."
    )
    tables = guided_tables()
    if tables is None:
        st.info("The answers of the reading test are not found.")
        return
    test_table, shares, third_table = tables
    st.subheader("Test sample (255 accounts: 135 positive, 120 clean)")
    st.dataframe(readable(test_table), hide_index=True, width="stretch")
    st.caption(
        "AUC-ROC from the reader's confidence. Precision depends on the share of "
        "positives, which is about half here and 1.2% in the whole bank."
    )
    if third_table is not None:
        st.subheader("Third sample of new accounts (seed 102, same strata)")
        st.dataframe(readable(third_table), hide_index=True, width="stretch")
        st.caption(
            "The two XGBoost readers use their out-of-fold scores; the ceiling model "
            "sees only the prompt's evidence and facts (notebook 17)."
        )
    st.subheader("Which accounts do the readers call suspicious? (test sample)")
    shares = shares.copy()
    shares.columns = [
        demo.reader_name(f"{model.split(':')[0]} {condition}", MODELS)
        for model, condition in shares.columns
    ]
    # Tall enough for every stratum, without a scroll bar.
    st.dataframe(shares.round(2), width="stretch", height=36 * (len(shares) + 1) + 3)
    st.caption("Share of the accounts in each stratum called suspicious.")


READING_TITLES = {
    pl.READ_ALONE: "Reading alone",
    pl.READ_SHOTS: "With the guide and examples",
}


@st.cache_data
def load_reading(directory: str, folder: str | None, results_file: str | None):
    path = None if folder is None else Path(folder)
    return demo.load_reading_set(Path(directory), path, results_file)


def show_reading_opinion(opinion: dict | None) -> None:
    if opinion is None:
        st.write("No stored answer.")
        return
    if opinion["decision"] is None:
        st.warning("The model gave no usable answer.")
        return
    # The word carries the decision; the colour only repeats it.
    colour = "red" if opinion["decision"] == sr.SUSPICIOUS else "green"
    st.markdown(f"Decision: :{colour}[**{opinion['decision'].replace('_', ' ')}**]")
    score = opinion.get("score")
    # The 0-100 score is asked only with the guide; reading alone shows confidence only.
    if score is None:
        st.metric("Confidence", f"{opinion['confidence']:.2f}", border=True)
    else:
        left, right = st.columns(2)
        left.metric("Confidence", f"{opinion['confidence']:.2f}", border=True)
        right.metric("Score 0-100", f"{score:.0f}", border=True)
    st.write(opinion["reason"] or "")


def show_readers_table(groups: pd.Series, results: pd.DataFrame) -> None:
    scores = demo.scores_directory()
    readers = fresh.ml_readers(
        groups.index,
        pd.read_parquet(
            scores / "transaction_reading.parquet", columns=["score", "flag"]
        ),
        pd.read_parquet(scores / "ceiling_facts_scores.parquet"),
    )
    if not results.empty:
        readers |= fresh.llm_readers(results[results["account"].isin(groups.index)])
    table = fresh.reader_table(groups, readers)
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.caption(
        "Balanced sample of equal strata, so the shares do not describe the "
        "population; small samples give wide intervals, which the table leaves out. "
        "XGBoost scores are out-of-fold: no account is scored by a model trained on it."
    )


RUN_STEPS = """
1. **Accounts.** The next accounts of a fresh pool (seed 102) that no live run has \
read, one stratum after another, so even a few mix positives and clean accounts. \
Their prompts were built beforehand from the transactions, with no label.
2. **XGBoost.** The stored out-of-fold score of the frozen transaction model. The \
reader never sees it; it is shown below for comparison.
3. **Local LLMs.** Each chosen model reads every account twice, alone and with the \
guide, through Ollama on this machine. Each answer appears below as it comes.
4. **Summary.** When the run ends, every reader is compared on all live accounts so \
far: AUC-ROC, recall and false alarm rate.
"""


@st.cache_data
def pool_cases(folder: str) -> pd.DataFrame:
    return demo.reading_cases(pd.read_parquet(Path(folder) / "prompts.parquet"))


def live_view(folder: Path, cases: pd.DataFrame, was_running: bool) -> None:
    """The run as it stands: status, answers so far, the log and, at the end, the
    summary. Reruns itself every two seconds while the run works."""
    status, last = demo.fresh_run_status(folder, demo.LIVE_LOG)
    if was_running and status != "running":
        # The run ended: the whole page reruns once, so the Start button returns.
        st.rerun(scope="app")
    answers = fresh.read_live_answers(folder / fresh.LIVE_ANSWERS)
    expected = demo.expected_answers(demo.live_plan(folder))
    if status == "not started":
        st.info(
            "No live run yet. Choose the models and the number of accounts, then Start."
        )
        return
    if status == "running":
        done = f"{len(answers)} answers so far" + (
            f" of {expected}" if expected else ""
        )
        st.progress(min(len(answers) / expected, 1.0) if expected else 0.0, done)
    elif status == "finished":
        st.success(f"The last run finished: {len(answers)} answers.")
    else:
        st.error(f"The last run failed: {last}")

    st.markdown("**Answers of this run, newest first**")
    if answers.empty:
        st.write("No answer yet: the model may still be loading.")
    else:
        table = demo.live_answer_table(answers, cases, MODELS)
        st.dataframe(table, hide_index=True, width="stretch")
        accounts = answers["account"].drop_duplicates()
        ml = demo.ml_view(accounts, demo.scores_directory())
        names = cases.set_index("account")["case"]
        st.markdown("**XGBoost on the same accounts (never shown to the reader)**")
        st.dataframe(
            pd.DataFrame(
                {
                    "case": accounts.map(names).fillna(accounts).to_numpy(),
                    "XGBoost score": ml["score"].to_numpy(),
                    "flagged at its threshold": np.where(ml["flag"], "yes", "no"),
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "XGBoost score": st.column_config.NumberColumn(format="%.4f")
            },
        )
    with st.expander("Log of the run", expanded=status == "running"):
        st.code(demo.log_tail(folder) or "(empty)", language=None)

    summary = folder / demo.LIVE_SUMMARY
    if status == "finished" and summary.exists():
        st.markdown("**Summary: every reader on all live accounts so far**")
        table = pd.read_csv(summary).round(3)
        table["reader"] = table["reader"].map(lambda r: demo.reader_name(r, MODELS))
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "A few accounts give wide intervals. The same answers can be opened one "
            "by one on the Guided reading page, sample 'New accounts, seed "
            f"{demo.POOL_SEED} ({demo.POOL_SIZE})'."
        )


def run_page() -> None:
    st.title("Run new accounts")
    st.write(
        "Sends accounts that the system has never read through the pipeline now, and "
        "shows each step, each answer and where the results are written."
    )
    directory = demo.reading_directory()
    folder = directory / fresh.run_name(demo.POOL_SEED, demo.POOL_SIZE)
    if not (folder / "prompts.parquet").exists():
        st.info(
            "The pool's prompts are not built yet. Run `scripts/run_fresh.py --seed "
            f"{demo.POOL_SEED} --size {demo.POOL_SIZE} --prepare` first."
        )
        return
    st.markdown(RUN_STEPS)

    status, _ = demo.fresh_run_status(folder, demo.LIVE_LOG)
    left, middle, right = st.columns([2, 1, 1])
    models = left.multiselect(
        "Models", list(MODELS), default=["gemma4:latest"], format_func=MODELS.get
    )
    default = 20 if models == ["gemma4:latest"] else 5
    accounts = middle.number_input(
        "Accounts", min_value=1, max_value=60, value=default, step=1
    )
    if models:
        right.metric(
            "Expected time", f"{demo.live_minutes(int(accounts), models):.1f} min"
        )
    if st.button(
        "Start", type="primary", disabled=not models or status == "running", key="start"
    ):
        demo.start_fresh_run(
            directory,
            demo.POOL_SEED,
            demo.POOL_SIZE,
            models,
            accounts=int(accounts),
            live=True,
        )
        load_reading.clear()
        st.rerun()

    st.subheader("The run")
    cases = pool_cases(str(folder))
    running = status == "running"
    st.fragment(run_every=2 if running else None)(live_view)(folder, cases, running)

    st.subheader("Where the results go")
    st.dataframe(demo.live_outputs(folder), hide_index=True, width="stretch")


def guided_page() -> None:
    st.title("Guided reading")
    st.write(
        "Each account is read twice by each model, without any ML output: alone, and "
        "with the facts beside the other accounts of the bank, a short guide and four "
        "solved examples. The answers are stored, so the page works with the local "
        "model switched off; a live call can be made on the demo cases."
    )
    directory = demo.reading_directory()
    sets = demo.reading_sets(directory)
    if not sets:
        st.info("No reading results found. Run `scripts/run_reading.py` first.")
        return
    name = st.sidebar.selectbox("Sample", list(sets))
    folder, results_file = sets[name]
    prompts, results = load_reading(
        str(directory), None if folder is None else str(folder), results_file
    )
    if folder is not None and results.empty:
        st.info("This sample has no answers yet.")
        return
    cases = demo.reading_cases(prompts)
    if not results.empty:
        cases = cases[cases["account"].isin(results["account"])]
    groups = cases.set_index("account")["group"]
    showcase = demo.demo_cases(results, groups)
    only_demo = st.sidebar.toggle(
        f"Demo cases only ({len(showcase)})", value=not showcase.empty
    )
    research = st.sidebar.toggle("Research view", value=False, key="reading_research")
    visible = cases[cases["account"].isin(showcase.index)] if only_demo else cases
    if visible.empty:
        visible = cases
    gaps = demo.cache_gaps(directory, prompts, visible["account"], MODELS)
    if gaps:
        st.sidebar.warning(f"{gaps} stored answers are missing for these cases.")
    else:
        st.sidebar.success("Every answer of these cases is stored.")

    chosen = st.selectbox("Case", visible["case"].tolist())
    case = visible[visible["case"] == chosen].iloc[0]
    if research:
        kind = "involved" if demo.is_involved(case["group"]) else "clean"
        rule = showcase.get(case["account"])
        st.info(
            f"Stratum: {case['group']} ({kind})"
            + (f". Demo rule: {rule}." if rule else ".")
        )

    alone_prompt = demo.case_prompt(prompts, case["account"], pl.READ_ALONE)["prompt"]
    guided_prompt = demo.case_prompt(prompts, case["account"], pl.READ_SHOTS)["prompt"]
    parts = demo.split_prompt(alone_prompt)
    sections = demo.guided_sections(guided_prompt)

    st.subheader("Evidence")
    show_summary(demo.parse_summary(parts["summary"]))
    st.dataframe(
        parts["transactions"].rename(columns=demo.TABLE_LABELS),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Facts beside the other accounts of the bank"):
        st.markdown("\n".join(sections["facts"]))
        st.markdown("\n".join(sections["comparison"]))
    with st.expander("The guide"):
        st.text(sections["guide"])
    with st.expander("The four solved examples"):
        for example in sections["examples"]:
            st.text(example)

    st.subheader("XGBoost (never shown to the reader)")
    ml = demo.ml_view([case["account"]], demo.scores_directory()).iloc[0]
    left, right = st.columns(2)
    left.metric("Transaction model score", f"{ml['score']:.3f}", border=True)
    right.metric("Flagged at its threshold", "yes" if ml["flag"] else "no", border=True)

    st.subheader("Second reader")
    live_set = results_file == demo.LIVE_RESULTS
    live_allowed = case["account"] in showcase.index and not live_set
    for column, model in zip(st.columns(len(MODELS)), MODELS, strict=True):
        with column:
            st.markdown(f"#### {MODELS[model]}")
            stored = {}
            for condition, prompt in [
                (pl.READ_ALONE, alone_prompt),
                (pl.READ_SHOTS, guided_prompt),
            ]:
                st.markdown(f"**{READING_TITLES[condition]}**")
                if live_set:
                    stored[condition] = demo.result_opinion(
                        results, case["account"], condition, model
                    )
                else:
                    stored[condition] = demo.cached_opinion(directory, prompt, model)
                show_reading_opinion(stored[condition])
            change = demo.guide_change_text(
                *(o["decision"] if o else None for o in stored.values())
            )
            if change:
                st.info(change)
            if not live_allowed:
                continue
            if st.button("Ask this model live", key=f"live {model}"):
                try:
                    with st.spinner("The local model is reading the case..."):
                        for condition, prompt in [
                            (pl.READ_ALONE, alone_prompt),
                            (pl.READ_SHOTS, guided_prompt),
                        ]:
                            live = demo.live_check(directory, prompt, model)
                            st.markdown(f"**Live: {READING_TITLES[condition]}**")
                            show_reading_opinion(live)
                            st.caption(
                                f"{live['seconds']:.1f} s. "
                                + demo.same_answer_text(stored[condition], live)
                                + (
                                    ""
                                    if live["same_version"]
                                    else " The installed model is not the version "
                                    "that gave the stored answers."
                                )
                            )
                except OSError as error:
                    st.error(
                        f"The local model could not be reached ({error}). The "
                        "stored answers above are what it said when the sample was "
                        "read."
                    )

    if research:
        st.subheader("How the readers did on this sample")
        show_readers_table(groups, results)


@st.cache_data(ttl=3600, show_spinner="Counting the tests (a few seconds)...")
def count_tests() -> int | None:
    return demo.collected_tests()


def checks_page() -> None:
    st.title("How it is checked")
    st.write(
        "What every number in this demo passed through, and what each step guards "
        "against. The details are in the notebooks."
    )
    st.graphviz_chart(demo.PIPELINE_DOT)
    st.table(demo.assurance_table().set_index("What we did"))
    left, right = st.columns(2)
    functions = demo.test_count()
    collected = count_tests()
    if collected is None:
        left.metric("Test functions", functions, border=True)
    else:
        left.metric("Tests", collected, border=True)
        left.caption(
            f"{functions} test functions; a function with several inputs counts once "
            "per input."
        )
    mutations = demo.mutation_summary(demo.scores_directory())
    if mutations is None:
        right.metric("Planted mistakes caught", "not run yet", border=True)
    else:
        right.metric(
            "Planted mistakes caught",
            f"{mutations['caught']} of {mutations['total']}",
            border=True,
        )
        st.caption(f"Last full mutation run: {mutations['date']}.")
        if mutations["missed"]:
            st.warning("Missed: " + "; ".join(mutations["missed"]))
    st.subheader("The rule for the guided reading, fixed before the results")
    st.write(
        "Main measure: AUC-ROC of the reader's confidence, against reading alone on "
        "the same accounts, with a stratified paired bootstrap (2,000 draws). The "
        "condition helped if the 95% interval of the change lies above 0. The "
        "condition was chosen on the development sample; the test sample was read "
        "once, in that condition (notebooks 15 and 16)."
    )


if page == "Guided reading":
    guided_page()
elif page == "Case review":
    review_page()
elif page == "Run new accounts":
    run_page()
elif page == "Experiment":
    experiment_page()
else:
    checks_page()
