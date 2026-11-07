"""The demo: an analyst reviews flagged accounts with an LLM second reader.

Run from the project root:

    uv run streamlit run app/streamlit_app.py
"""

import pandas as pd
import streamlit as st

from skeptical_second_reader import demo
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
page = st.sidebar.radio("Page", ["Case review", "Experiment"], key="page")
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


if page == "Case review":
    review_page()
else:
    experiment_page()
