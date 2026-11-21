"""The demo: an analyst reviews flagged accounts with an LLM second reader.

Run from the project root:

    uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from skeptical_second_reader import demo, fresh
from skeptical_second_reader import pipeline as pl
from skeptical_second_reader import second_reader as sr

MODELS = {"gemma4:latest": "Gemma 4", "qwen3.5:9b-q4_K_M": "Qwen 3.5 9B"}
CONDITION_NAMES = {
    sr.K2: "True ML output",
    sr.K1: "No ML output (the reader alone)",
    sr.K3_D33: "Corrupted ML output, 33%",
    sr.K3_D66: "Corrupted ML output, 66%",
    sr.K3_D100: "Corrupted ML output, 100%",
    sr.SCORE_ONLY: "Corrupted ML output, 100%, no reasons",
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
PAGES = ["Guided reading", "Case review", "Experiment", "How it is checked"]
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
            st.warning("This ML output is deliberately corrupted for the experiment.")
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


def experiment_page() -> None:
    st.title("Experiment")
    st.write(
        "Each account is read alone and with a true or corrupted ML output. The "
        "evidence never changes. Among accounts the reader decides correctly alone, "
        "how many turn wrong in each condition? The table pools all groups: where the "
        "ML model is right a corrupted output is wrong, where it is wrong a corrupted "
        "output is right, so notebook 11 gives these shares per group."
    )
    progress = demo.progress_table(directory)
    expected = len(prompts)
    if progress.empty:
        st.info("No finished results yet.")
        return
    for entry in progress.itertuples():
        st.progress(
            min(entry.answers / expected, 1.0),
            f"{entry.model}: {entry.answers} of {expected}",
        )
    if (progress["answers"] < expected).any() or len(progress) < len(MODELS):
        st.warning("The run is not finished: numbers below are partial.")
    results = pd.read_parquet(directory / demo.RESULTS)
    reading = pd.read_parquet(directory / demo.READING, columns=["label"])
    table = demo.dose_table(results, reading)
    table["condition"] = table["condition"].map(CONDITION_NAMES)
    st.dataframe(table, hide_index=True, width="stretch")
    st.bar_chart(
        table, x="condition", y="share", color="model", stack=False, horizontal=True
    )
    st.caption(
        "Sample of equal groups, so these shares do not describe the population. "
        "The prepared analysis with intervals is in notebook 11."
    )


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
    left, right = st.columns(2)
    left.metric("Confidence", f"{opinion['confidence']:.2f}", border=True)
    score = opinion.get("score")
    right.metric(
        "Score 0-100", "not asked" if score is None else f"{score:.0f}", border=True
    )
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


def run_pipeline_panel(directory: Path) -> None:
    """Optional: read accounts the system has never seen, now."""
    folder = directory / fresh.run_name(demo.POOL_SEED, demo.POOL_SIZE)
    if not (folder / "prompts.parquet").exists():
        return
    with st.sidebar.expander("Run the pipeline live"):
        st.write(
            "Takes the next accounts of a fresh pool that no live run has read, "
            "scores them with XGBoost and asks the local models now. The pool's "
            "prompts are built beforehand, so no data has to load."
        )
        models = st.multiselect(
            "Models", list(MODELS), default=["gemma4:latest"], format_func=MODELS.get
        )
        default = 20 if models == ["gemma4:latest"] else 5
        accounts = st.number_input(
            "Accounts", min_value=1, max_value=60, value=default, step=1
        )
        if models:
            minutes = demo.live_minutes(int(accounts), models)
            st.caption(f"About {minutes:.1f} minutes.")
        status, last = demo.fresh_run_status(folder, demo.LIVE_LOG)
        if status == "running":
            st.write(f"Running: {last}")
        elif status in ("finished", "failed"):
            st.write(f"Last run **{status}**. {last}")
        if st.button("Start", disabled=not models or status == "running"):
            demo.start_fresh_run(
                directory,
                demo.POOL_SEED,
                demo.POOL_SIZE,
                models,
                accounts=int(accounts),
                live=True,
            )
            st.rerun()
        if st.button("Refresh"):
            load_reading.clear()
            st.rerun()


def guided_page() -> None:
    st.title("Guided reading")
    st.write(
        "Each account is read twice by each model, without any ML output: alone, and "
        "with the facts beside the other accounts of the bank, a short guide and four "
        "solved examples. The answers are stored, so the page works with the local "
        "model switched off; a live call can be made on the demo cases."
    )
    directory = demo.reading_directory()
    run_pipeline_panel(directory)
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


def checks_page() -> None:
    st.title("How it is checked")
    st.write(
        "What every number in this demo passed through, and what each step guards "
        "against. The details are in the notebooks."
    )
    st.graphviz_chart(demo.PIPELINE_DOT)
    st.table(demo.assurance_table().set_index("What we did"))
    left, right = st.columns(2)
    left.metric("Test functions", demo.test_count(), border=True)
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
elif page == "Experiment":
    experiment_page()
else:
    checks_page()
