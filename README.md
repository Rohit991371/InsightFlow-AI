# InsightFlow AI

**Autonomous Multi-Agent Data Analyst → AI Analytics Consultant**

Upload a CSV/Excel dataset and get back a data quality assessment, statistical
insights, visual dashboards with plain-language explanations, proactive
analysis recommendations, a live chart-on-demand interface, a conversational
data Q&A chat, and a downloadable executive PDF report — no SQL, Python, or
data analysis expertise required.

Built with **LangGraph** orchestrating seven specialized agents in a linear
pipeline, plus two on-demand interactive agents driven from the Streamlit UI,
with **Groq (Llama 3.3 70B, free tier)** powering every language/reasoning
step and clean deterministic fallbacks everywhere so the app still works
fully without an API key.

---

## Architecture

```
User → Upload CSV/Excel → Orchestrator (LangGraph)

  Data Cleaning → Statistician → Visualization
        → Chart Explanation → Recommendation
        → Business Analyst → Report Writer → Executive Report.pdf

Interactive layer (live, on-demand from the Streamlit UI):

  User chart request → Interactive Analytics Agent → Chart Explanation Agent
                                                            │
  User question      → Dataset Chat Agent ──────────────────┴──→ stored in
                                                                   session state,
                                                                   folded into
                                                                   the PDF on
                                                                   "Refresh Report"
```

> **Note:** the original vision doc shows agents 1–3 as parallel branches.
> This is implemented as a **linear** LangGraph since each step's output is
> independent and a linear graph is simpler to debug. True parallelism is a
> natural next step if performance becomes a concern (see
> [Future Work](#future-work)).

### Pipeline Agents (run once per uploaded dataset)

| # | Agent | File | Responsibility |
|---|-------|------|-----------------|
| 1 | Data Cleaning | `agents/cleaner.py` | Missing values, duplicates, empty/near-empty columns, mistyped numeric-as-text columns |
| 2 | Statistician | `agents/statistician.py` | Descriptive stats, correlation matrix, top categories, IQR-based outlier detection |
| 3 | Visualization | `agents/visualizer.py` | Histograms, bar charts, pie chart, trend line, correlation heatmap — each tagged with `chart_metadata` describing what it plots |
| 4 | Chart Explanation | `agents/chart_explainer.py` | For every chart: why it exists, variable definitions, what's compared, key observations, business interpretation, recommended actions |
| 5 | Recommendation | `agents/recommender.py` | Proactively suggests up to 5 follow-up analyses (grounded in real correlations/outliers/trends), each with a ready-to-run chart request |
| 6 | Business Analyst | `agents/analyst.py` | Translates technical findings into plain-language insights + recommendations |
| 7 | Report Writer | `agents/report_writer.py` | Assembles everything — including interactive session content — into a polished PDF via ReportLab |

### Interactive Agents (live, on-demand)

| Agent | File | Responsibility |
|-------|------|-----------------|
| Interactive Analytics | `agents/interactive_analytics.py` | Parses a natural-language chart request (LLM or rule-based) into a structured spec, validates columns with fuzzy-match suggestions, renders the chart (scatter, line, bar, histogram, pie, box, heatmap, area) |
| Dataset Chat | `agents/dataset_chat.py` | Answers natural-language questions by generating a single pandas expression, executing it through a sandboxed evaluator, and explaining the result in business language |

### Supporting Infrastructure

| File | Responsibility |
|------|-----------------|
| `utils/llm_client.py` | Shared Groq call wrapper — JSON parsing, markdown-fence stripping, consistent `LLMUnavailableError` handling across every agent that uses an LLM |
| `utils/sandbox.py` | Restricted execution environment for LLM-generated pandas expressions (Dataset Chat Agent) — see [Security](#security-llm-generated-code-execution) below |

---

## Security: LLM-Generated Code Execution

The Dataset Chat Agent asks an LLM to produce a pandas expression that
answers a user's question, then runs it against the real dataframe. This is
**never** done with raw `eval()`/`exec()` on untrusted text. Instead,
`utils/sandbox.py` enforces:

- **`eval()`, not `exec()`** — the LLM is only ever asked for a single
  *expression*, not a script. This makes statements, imports, assignments,
  loops, and `def`/`class` syntactically impossible from the start.
- **AST validation before execution** — the parsed expression is walked and
  rejected if it references any name outside an explicit allowlist
  (`df`, `pd`, `np`, a few literals), or accesses any attribute containing
  a dunder, `to_csv`/`to_sql`/etc., `query`, `exec`, `eval`, `compile`,
  `open`, `os`, `sys`, or `subprocess`.
- **No builtins leak** — `__builtins__` is replaced wholesale with a tiny
  safe subset (`len`, `min`, `max`, `sum`, `round`, `abs`, `sorted`, basic
  type constructors). Nothing else is reachable.
- **Wall-clock timeout** (5s, worker thread) so a pathological expression
  can't hang the app.
- **Result size cap** (200 rows) before any result is returned for display
  or further LLM prompting.

This blocks the standard sandbox-escape techniques (e.g.
`().__class__.__bases__[0].__subclasses__()`), filesystem access, and data
exfiltration — see `tests/test_pipeline.py::TestSandboxSecurity` for the
full set of payloads tested against it.

---

## Project Structure

```
InsightFlow-AI/
│
├── app.py                      # Streamlit UI — pipeline + live chat + custom charts
├── config.py
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── cleaner.py
│   ├── statistician.py
│   ├── visualizer.py
│   ├── chart_explainer.py
│   ├── recommender.py
│   ├── interactive_analytics.py
│   ├── dataset_chat.py
│   ├── analyst.py
│   └── report_writer.py
│
├── workflow/
│   └── graph.py                # LangGraph orchestrator (7-node linear pipeline)
│
├── utils/
│   ├── data_loader.py           # CSV/Excel ingestion
│   ├── llm_client.py            # Shared Groq call wrapper
│   ├── sandbox.py               # Restricted pandas-expression evaluator
│   └── pdf_generator.py         # ReportLab PDF assembly (incl. interactive content)
│
├── tests/
│   └── test_pipeline.py        # pytest suite (44 tests)
│
├── uploads/                    # uploaded datasets land here
├── charts/                     # generated chart PNGs (auto + custom)
└── reports/                    # generated PDF reports
```

---

## Setup

```bash
# 1. Clone / enter the project directory
cd InsightFlow-AI

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure your Groq API key for LLM-powered insights,
#    explanations, recommendations, custom charts, and chat
cp .env.example .env
# then edit .env and paste your key from https://console.groq.com
```

> **No API key? No problem.** Every LLM-powered agent — Business Analyst,
> Chart Explanation, Recommendation, Interactive Analytics, Dataset Chat —
> falls back to a deterministic, rule-based path if `GROQ_API_KEY` is
> missing or a call fails. Nothing in the pipeline hard-fails; you'll see a
> `"source": "fallback_template"` or `"source": "rule_based"` flag instead
> of `"source": "llm"` in the relevant output.

---

## Running the App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`):

1. Upload a CSV/Excel file and (optionally) paste your Groq key in the sidebar
2. Click **Run Analysis** — runs the full 7-agent pipeline
3. Browse the auto-generated charts, each with its own explanation
4. Click **Generate** on any recommended analysis, or type your own request
   under **Request a Custom Chart**
5. Ask questions about your data under **Ask Your Data**
6. Click **Refresh Report with Custom Charts & Q&A**, then **Download PDF
   Report** — the PDF includes everything generated during the session

---

## Running the Pipeline Programmatically

```python
import pandas as pd
from workflow.graph import run_pipeline

df = pd.read_csv("uploads/sales.csv")
result = run_pipeline(df, dataset_name="sales.csv")

print(result["business_insights"]["executive_summary"])
print(result["recommendations"]["recommendations"])  # proactive analysis suggestions
print(result["report_path"])  # path to generated PDF
```

The returned `result` dict contains every agent's output: `dataset_info`,
`cleaning_report`, `stats_report`, `charts` (including `chart_metadata`),
`chart_explanations`, `recommendations`, `business_insights`, and
`report_path`.

To use the interactive agents outside Streamlit:

```python
from agents.interactive_analytics import run_interactive_analytics_agent
from agents.dataset_chat import run_dataset_chat_agent

chart = run_interactive_analytics_agent("scatter plot between age and salary", df)
answer = run_dataset_chat_agent("Which product generated the highest revenue?", df)
```

---

## Running Tests

```bash
pip install pytest   # already in requirements.txt
pytest tests/ -v
```

44 tests cover the data loader, every agent individually (including the four
new Enhancement Phase agents), the sandbox's security boundary against a set
of real exploit payloads, PDF generation with and without interactive
content, the fallback path when no Groq key is set, and full end-to-end
pipeline runs on both a messy and a minimal dataset.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangGraph |
| LLM | Groq — Llama 3.3 70B (free tier) |
| Analytics | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Report | ReportLab |
| Conversational layer | Sandboxed pandas-expression evaluation (custom, not a 3rd-party agent framework) |

---

## Design Notes / Known Limitations

- **Linear graph, not parallel.** Agents 1–3 (and the explanation/recommendation
  steps after them) run sequentially even though parts of their work are
  independent. Fine for typical CSV sizes; revisit if processing very large
  files or many datasets concurrently.
- **Charts skip gracefully when data doesn't support them.** No datetime
  column → no trend line. No 2–8-cardinality categorical column → no pie
  chart. Fully-empty numeric/categorical columns are excluded everywhere.
- **PDF embeds raster PNGs**, not vector charts — keeps ReportLab simple, at
  a small cost to print quality at extreme zoom.
- **Dataset Chat is read-only by design.** The sandbox explicitly blocks
  `to_csv`/`to_sql`/etc. and any expression that could mutate or export data —
  it answers questions, it doesn't modify your file.
- **Rule-based fallbacks are intentionally narrow.** Without an API key, the
  Dataset Chat Agent handles common patterns (average/max/min, "how many X
  more than Y", "which category has the highest Y", top-N, row counts) but
  won't handle genuinely open-ended or judgment-based questions ("which
  region is underperforming?") — it fails cleanly with a suggestion to
  rephrase rather than guessing.
- **Recommendation Agent avoids duplicates** by checking already-generated
  chart metadata before suggesting new analyses, in both the LLM and
  rule-based paths.

## Future Work

- True parallel branching in LangGraph for independent pipeline steps
- Forecasting agent (e.g. simple trend extrapolation or Prophet)
- Multi-file / multi-sheet Excel support
- Persist chat/custom-chart history across page reloads (currently
  session-scoped only)
- Caching layer so re-running on the same file skips redundant agent calls
- Auth + per-user upload history if deployed beyond local/demo use
