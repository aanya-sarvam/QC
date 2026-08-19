"""
4-Way Prompt Comparison — Original IGR · Prompt 1 · Expert-corrected · Prompt 2
Run locally:  streamlit run app_compare4.py

For each deed, shows every field across four columns:
  1. Original IGR metadata that was sent to Gemini as the grounding anchor
  2. Prompt 1 (prompt.py) Odia extraction
  3. Expert-corrected Odia (ground truth)
  4. Prompt 2 (prompt_v2.py) Odia extraction
plus a match verdict for Prompt 1 and Prompt 2 against the expert value.

Expects these three files in the same folder (or upload from the sidebar):
  vertex_10_corrected.json   (DB export: odia_value = expert truth)
  old_outputs_10.jsonl       (prompt.py output, one JSON per deed)
  new_outputs.json           (prompt_v2.py output)
"""
import os
import tempfile

import pandas as pd
import streamlit as st

from reconcile4 import build

st.set_page_config(page_title="4-Way Prompt Comparison", layout="wide")
st.title("🔬 Prompt v1 vs v2 — 4-Way Deed Comparison")
st.caption(
    "Original IGR metadata · Prompt 1 (prompt.py) · Expert-corrected ground truth · "
    "Prompt 2 (prompt_v2.py). Verdicts compare each prompt against the expert value."
)


def _save_upload(uploaded, suffix):
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return tmp.name


with st.sidebar:
    st.header("Data source")
    st.caption("Defaults to bundled files — upload to override.")
    up_export = st.file_uploader("vertex_10_corrected.json", type="json", key="exp")
    up_old = st.file_uploader("old_outputs_10.jsonl", type=["jsonl", "json"], key="old")
    up_new = st.file_uploader("new_outputs.json", type="json", key="new")

export_path = _save_upload(up_export, ".json") or "vertex_10_corrected.json"
old_path = _save_upload(up_old, ".jsonl") or "old_outputs_10.jsonl"
new_path = _save_upload(up_new, ".json") or "new_outputs.json"

for p, name in [(export_path, "export"), (old_path, "old outputs"), (new_path, "new outputs")]:
    if not os.path.exists(p):
        st.warning(f"Missing {name} file ({p}). Upload it from the sidebar.", icon="⚠️")
        st.stop()

df, deed_types = build(export_path, old_path, new_path)
if df.empty:
    st.error("No comparable rows produced — check the three files share deed numbers.", icon="🚫")
    st.stop()

# ---------------- overall summary ----------------
GOOD = {"match", "both-blank", "deed_type", "extra"}


def coverage_recall(side):
    # recall = fraction of expert-valued fields the prompt matched
    has_truth = df["Expert corrected (Odia)"].str.strip() != ""
    sub = df[has_truth]
    matched = sub[side].isin({"match", "deed_type"}).sum()
    covered = (sub[side] != "missing").sum()
    n = len(sub)
    return covered / n, matched / n, n


c1_cov, c1_rec, n1 = coverage_recall("P1")
c2_cov, c2_rec, n2 = coverage_recall("P2")

st.subheader("Overall (fields where the expert entered a value)")
m1, m2, m3 = st.columns(3)
m1.metric("Coverage — Prompt 1", f"{c1_cov*100:.1f}%")
m1.metric("Coverage — Prompt 2", f"{c2_cov*100:.1f}%", f"{(c2_cov-c1_cov)*100:+.1f} pp")
m2.metric("Match — Prompt 1", f"{c1_rec*100:.1f}%")
m2.metric("Match — Prompt 2", f"{c2_rec*100:.1f}%", f"{(c2_rec-c1_rec)*100:+.1f} pp")
fixed = ((df["P1"] == "missing") & (df["P2"].isin({"match", "deed_type"}))).sum()
broke = ((df["P1"].isin({"match", "deed_type"})) & (df["P2"] == "mismatch")).sum()
m3.metric("Fields fixed by v2", int(fixed))
m3.metric("Fields regressed by v2", int(broke))
st.caption(
    "Coverage = produced a value where the expert had one (the client's "
    "'not found' complaint). Match = value agrees with the expert after "
    "Odia-aware date/number/spelling folding."
)

# ---------------- per-deed table ----------------
st.subheader("Per-deed field comparison")
deeds = sorted(df["deed"].unique())
labels = [f"{d}  ({deed_types.get(d,'')})" for d in deeds]
choice = st.selectbox("Select deed", options=list(range(len(deeds))),
                       format_func=lambda i: labels[i])
deed = deeds[choice]
ddf = df[df["deed"] == deed].copy()

show = ddf[[
    "group", "field",
    "Original IGR (sent to Gemini)",
    "Prompt 1 (Odia)", "P1",
    "Expert corrected (Odia)",
    "Prompt 2 (Odia)", "P2",
]].rename(columns={"group": "Section", "field": "Field", "P1": "P1?", "P2": "P2?"})

STATUS_EMOJI = {
    "match": "✅ match", "mismatch": "❌ differs", "missing": "⬜ not found",
    "both-blank": "· blank", "deed_type": "≈ category", "extra": "➕ extra",
}
show["P1?"] = show["P1?"].map(lambda s: STATUS_EMOJI.get(s, s))
show["P2?"] = show["P2?"].map(lambda s: STATUS_EMOJI.get(s, s))


def color_verdict(val):
    if val.startswith("✅"):
        return "background-color:#1b5e20;color:white;"
    if val.startswith("❌"):
        return "background-color:#b71c1c;color:white;"
    if val.startswith("⬜"):
        return "background-color:#e65100;color:white;"
    if val.startswith("➕"):
        return "background-color:#0d47a1;color:white;"
    return ""


styled = show.style.applymap(color_verdict, subset=["P1?", "P2?"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=560)

st.caption(
    "✅ match · ❌ differs from expert · ⬜ not found (expert had a value, prompt blank) · "
    "➕ extra (prompt found something the expert left blank) · ≈ category (deed_type: "
    "English label vs on-page Odia term — expected)."
)

with st.expander("What changed between Prompt 1 and Prompt 2?"):
    st.markdown(
        "- **Write-through / conflict fix (self-check rule).** v1 told Gemini to "
        "*blank the field* whenever its reading disagreed with the IGR metadata. v2 "
        "splits that into (a) genuine mis-location → re-scan, and (b) correct location "
        "but conflicting value → keep the value, mark found, lower confidence, and note "
        "the conflict. Stops correct-but-conflicting readings from being discarded.\n"
        "- **Address extraction.** v2 adds explicit guidance to read the whole "
        "multi-line address block (village / PO / PS / district), transcribe partial "
        "legible text instead of blanking, and match it to the correct party.\n"
        "- **Plot boundary as a first-class field.** v2 documents `property_boundary` "
        "(north/south/east/west) as a list field whose values may be names, landmarks, "
        "plots, or places — so boundary sides get extracted rather than skipped."
    )
