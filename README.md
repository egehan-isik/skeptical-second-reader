# Skeptical Second Reader

[![CI](https://github.com/egehan-isik/skeptical-second-reader/actions/workflows/ci.yml/badge.svg)](https://github.com/egehan-isik/skeptical-second-reader/actions/workflows/ci.yml)

**LLM-Assisted Decision Support for Financial Risk Analysis**

## Overview

A machine learning model flags suspicious accounts in a synthetic banking dataset.
A large language model then reviews each case as a second reader and gives its own decision.
The project builds this pipeline as a working prototype and uses it to test how far the language model can be trusted.

## Research question

Does the language model check the evidence, or does it simply follow the machine learning output, even when that output is wrong?

To test this, the language model receives correct ML outputs and deliberately corrupted ones with increasing levels of error, and we measure how often it changes its decision.

## Status

Work in progress. Graduation Project I, Computer Engineering, Manisa Celal Bayar University (2026–27).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data

No data is stored in this repository.

The project uses the **IBM Transactions for Anti-Money Laundering** dataset (HI-Small), a synthetic dataset released under the CDLA-Sharing-1.0 license:
<https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>

Download it yourself and place the files under `data/raw/` (this folder is git-ignored).

## License

The code is released under the [MIT License](LICENSE).
The license covers the code in this repository only; the dataset keeps its own license.
