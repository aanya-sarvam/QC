"""
diff_logic.py — reconciles the deed-validator ground truth export
(ground_truth.csv, section/label keyed, comma-joined multi-item strings)
against a fresh Gemini realtime run (realtime_fields.csv, field_id/attr/
item_index keyed) so the two can be compared field-by-field per deed.

Canonical field vocabulary follows the Gemini side (field_id[.attr]),
since that's what prompt.py needs to be refined against.
"""
import re
from difflib import SequenceMatcher
import pandas as pd

# section (lowercased, "(n)" suffix stripped) + label (lowercased) -> canonical field
LABEL_MAP = {
    ("deed details", "deed type"): "deed_type",
    ("deed details", "district"): "district",
    ("deed details", "office"): "office",
    ("deed details", "registration office"): "office",
    ("deed details", "registration date"): "registration_date",
    ("deed details", "presentation date"): "presentation_date",
    ("deed details", "consideration amount"): "consideration_amount",
    ("deed details", "old reg no"): "old_reg_no",
    ("deed details", "old registration no"): "old_reg_no",
    ("seller", "name"): "seller_details.name",
    ("seller", "relation name"): "seller_details.relation_name",
    ("seller", "address"): "seller_details.address",
    ("buyer", "name"): "buyer_details.name",
    ("buyer", "relation name"): "buyer_details.relation_name",
    ("buyer", "address"): "buyer_details.address",
    ("properties", "village"): "property_details.village",
    ("properties", "khata"): "property_details.khata",
    ("properties", "plot"): "property_details.plot",
    ("properties", "area"): "property_details.area",
}

_STRIP_COUNT = re.compile(r"\s*\(\d+\)\s*$")


def _norm_section(section: str) -> str:
    s = _STRIP_COUNT.sub("", str(section or "")).strip().lower()
    if s.startswith("seller"):
        return "seller"
    if s.startswith("buyer"):
        return "buyer"
    if s.startswith("propert"):
        return "properties"
    return s


def canonical_field(section: str, label: str):
    key = (_norm_section(section), str(label or "").strip().lower())
    return LABEL_MAP.get(key)


def _clean(v):
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _norm_compare(v: str) -> str:
    """Loose normalization for equality checks: collapse whitespace/commas,
    drop trailing punctuation, case-fold. Not used for display, only for
    the match/mismatch verdict."""
    v = _clean(v)
    v = re.sub(r"\s+", " ", v)
    v = v.strip(" ,.")
    return v.lower()


# --- Aggressive content-equivalence normalization (readback channel) ---------
# Per the reviewer's rule: spelling variants, script differences, date/number
# FORMATTING, currency formatting, and accidental DB duplication are all NOT
# content mismatches — only a genuinely different value should be flagged red.

_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}
# spelling-variant folding for common Odia romanization pairs
_SPELL_FOLD = [
    ("sahoo", "sahu"), ("oo", "u"), ("v", "b"), ("w", "b"), ("z", "j"),
    ("ph", "f"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"),
    ("kh", "k"), ("ch", "c"), ("sh", "s"), ("ee", "i"), ("aa", "a"),
    ("ou", "u"), ("y", "i"),
]


_ODIA_DIGITS = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")


def _to_latin_digits(s: str) -> str:
    return s.translate(_ODIA_DIGITS)


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", _to_latin_digits(s))


# --- Odia spelling-variant tolerance (names / places) ------------------------
# The prompt itself explicitly tells Gemini that names and place names can
# legitimately have more than one valid spelling/transliteration for the SAME
# entity (matching dialectal/OCR variation in these deeds: retroflex vs dental
# consonants, nukta presence, long/short vowel matras, nasal clusters). We
# mirror that tolerance here so the viewer doesn't flag those as real content
# errors — only a genuinely different name/place should turn red.
def _fold_odia_variants(s: str) -> str:
    s = s.strip()
    s = s.replace("଼", "")                              # drop nukta diacritic
    s = s.replace("ଣ", "ନ").replace("ଳ", "ଲ")            # retroflex -> dental
    s = s.replace("ଶ", "ସ").replace("ଷ", "ସ")            # sibilant merge
    s = s.replace("ଵ", "ବ")                              # v/b merge
    s = re.sub("ଙ୍", "", s)                              # nasal cluster simplify
    s = re.sub("ଞ୍", "", s)
    s = re.sub("ଂ", "", s)                               # bare anusvara
    s = s.replace("ୌ", "ୋ").replace("ୋ", "ୁ").replace("ୂ", "ୁ")  # vowel matras
    s = s.replace("ୀ", "ି").replace("ଈ", "ଇ").replace("ଊ", "ଉ")
    return s


_ODIA_BLOCK = re.compile(r"[\u0B00-\u0B7F]")


def _is_odia_script(s: str) -> bool:
    return bool(_ODIA_BLOCK.search(s))


_TOKEN_SPLIT = re.compile(r"[\s,\-।]+")

# Last-resort alias table: canonical key for well-known, frequently-recurring
# Odisha place names, covering both Odia-script and Latin spellings/variants
# seen in this corpus. Used only when NEITHER side has a readback available
# to cross-check against (so fuzzy/script-fallback matching has nothing to
# work with) — a narrow, low-risk safety net, not a general transliterator.
_PLACE_ALIASES = {
    "angul": "angul", "anugul": "angul", "anugola": "angul", "angula": "angul",
    "ଅନୁଗୋଳ": "angul", "ଅନୁଗୁଳ": "angul", "ଅଙ୍ଗୁଳ": "angul", "ଆଙ୍ଗୁଲ": "angul",
}


def _place_alias(s: str) -> str | None:
    return _PLACE_ALIASES.get(_clean(s).strip().lower())

# Fields where spelling-variant / granularity tolerance applies (per prompt's
# own NAMES / PLACE NAMES matching rules). Exact-value fields (dates, amounts,
# khata/plot/old_reg_no) are handled separately and stay strict.
_FUZZY_FIELDS = {
    "district", "office", "deed_type",
}
_FUZZY_ATTRS = {"village", "name", "relation_name", "address", "area"}

_FUZZY_THRESHOLD = 0.55


def _fuzzy_equivalent(a: str, b: str, field: str) -> bool:
    """True if a and b are a tolerable spelling variant / granularity
    difference of the SAME name or place, per the prompt's own matching
    philosophy. Compares folded strings directly, and (for list-item fields
    with descriptive clauses) also checks token-level overlap so a short
    extracted core value (e.g. 'Kushakila') matches against a fuller
    descriptive clause that contains the same place/name embedded in it
    (e.g. 'mouza - Kushalika, thana - ...')."""
    fa, fb = _fold_odia_variants(a), _fold_odia_variants(b)
    if not fa or not fb:
        return False
    if SequenceMatcher(None, fa, fb).ratio() >= _FUZZY_THRESHOLD:
        return True
    # token-level: does any token of the longer clause resemble the shorter
    # value closely enough? (handles full-clause vs core-token granularity)
    longer, shorter = (fa, fb) if len(fa) >= len(fb) else (fb, fa)
    tokens = [t for t in _TOKEN_SPLIT.split(longer) if t]
    if len(tokens) > 1:
        best = max((SequenceMatcher(None, shorter, t).ratio() for t in tokens), default=0)
        if best >= _FUZZY_THRESHOLD:
            return True
    return False


def _try_date_key(v: str) -> str | None:
    """If v looks like a date (any common format), return a canonical
    YYYYMMDD-ish key so 07-Nov-2000 == 7th Nov 2000 == 7/11/00. Returns None
    if it doesn't look like a date."""
    s = _to_latin_digits(v).lower().strip()
    s = s.replace("।", "|")  # Odia danda used as a date separator in some deeds
    s = re.sub(r"\s+", "", s)  # drop internal spaces: '9 .10 .02' -> '9.10.02'
    # Month-name form: 7th nov 2000 / 07-nov-2000 / 26th day of march 03
    m = re.search(r"(\d{1,2}).*?(" + "|".join(_MONTHS) + r")\w*.*?(\d{2,4})", s)
    if m:
        d, mon, y = m.group(1), _MONTHS[m.group(2)[:3]], m.group(3)
        y = y[-2:]  # compare on last two year digits (deeds mix 2000/00)
        return f"{int(d):02d}{mon}{y}"
    # Numeric form: 22-2-2001 / 9.10.02 / 31-10-02 / 24/6/03 / 22|2|03
    m = re.fullmatch(r"(\d{1,2})[.\-/|](\d{1,2})[.\-/|](\d{2,4})", s)
    if m:
        d, mon, y = m.group(1), m.group(2), m.group(3)
        y = y[-2:]
        return f"{int(d):02d}{int(mon):02d}{y}"
    return None


def _content_key(v: str, field: str = "") -> str:
    """Canonical key for the 'is this genuinely different content?' verdict.
    Folds away spelling / script / date-format / currency-format / duplicate
    differences so only real content changes survive."""
    v = _clean(v)
    if v == "":
        return ""

    v = _to_latin_digits(v)  # Odia digits -> Latin so dates/numbers compare
    base = field.split(".")[0]
    attr = field.split(".")[1] if "." in field else ""

    # consideration_amount: one rupee figure. Strip grouping commas and any
    # trailing paise/decimal part (/-, /00, .00, -00) so 4,950 == 4950 and
    # 6000/- == 6000.00 == 6000.
    if base == "consideration_amount":
        s = _to_latin_digits(v)
        s = re.sub(r"[.,/\-]\s*0*\s*$", "", s)       # trailing separator+zeros
        s = re.sub(r"[.\-/]0{1,2}\b", "", s)          # embedded .00 / -00 paise
        d = re.sub(r"\D", "", s)
        d = d.lstrip("0") or "0"
        return d

    # other numeric-ish fields (khata, plot, old_reg_no): digits only, but
    # collapse comma-duplicated repeats (upstream dup) to a unique set
    if base == "old_reg_no" or attr in {"khata", "plot"}:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        digits = [_digits_only(p).lstrip("0") or "0" for p in parts]
        digits = [d for d in digits if d]
        uniq = sorted(set(digits))
        return "|".join(uniq)

    # date fields: canonical date key
    if attr == "" and base in {"registration_date", "presentation_date", "execution_date"}:
        dk = _try_date_key(v)
        if dk:
            return dk
        # fall through to text handling if unparseable

    # generic text (names, places, addresses, villages, areas): fold spelling,
    # drop script/punctuation, collapse duplicate comma-items
    parts = [p.strip() for p in v.split(",")]
    folded = []
    for p in parts:
        p = p.lower()
        p = re.sub(r"[^a-z0-9\u0b00-\u0b7f]", "", p)  # keep latin + odia block
        for a, b in _SPELL_FOLD:
            p = p.replace(a, b)
        if p:
            folded.append(p)
    uniq = sorted(set(folded))
    return "|".join(uniq)


def load_ground_truth(path_csv: str) -> pd.DataFrame:
    gt = pd.read_csv(path_csv)
    gt["field"] = gt.apply(lambda r: canonical_field(r["section"], r["label"]), axis=1)
    gt["deed_number"] = gt["deed_number"].astype(str)
    return gt


def load_realtime(path_csv: str) -> pd.DataFrame:
    rt = pd.read_csv(path_csv)
    rt["reg_no"] = rt["reg_no"].astype(str)
    rt["field"] = rt.apply(
        lambda r: r["field_id"] if _clean(r.get("attr")) == ""
        else f"{r['field_id']}.{r['attr']}", axis=1)
    if "latin_readback" not in rt.columns:
        rt["latin_readback"] = ""
    return rt


def _joined_realtime_value(rt_rows: pd.DataFrame, col: str) -> str:
    """Join a field's per-item_index rows (english_value or odia_text) in
    item_index order with ', ' — mirrors the DB's comma-joined group value."""
    ordered = rt_rows.sort_values("item_index")
    vals = [_clean(v) for v in ordered[col].tolist()]
    return ", ".join(vals)


def build_comparison(gt: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    """One row per (deed_number, canonical field): ground truth (English +
    Odia, as corrected by the expert) vs the fresh Gemini run's joined value,
    plus a match verdict for each channel."""
    rows = []
    deed_numbers = sorted(set(gt["deed_number"]) & set(rt["reg_no"]))
    for deed in deed_numbers:
        gt_d = gt[gt["deed_number"] == deed]
        rt_d = rt[rt["reg_no"] == deed]
        fields = sorted(set(f for f in gt_d["field"].tolist() if f))
        for field in fields:
            gt_row = gt_d[gt_d["field"] == field]
            if gt_row.empty:
                continue
            gt_row = gt_row.iloc[0]
            rt_rows = rt_d[rt_d["field"] == field]

            gt_en = _clean(gt_row["corrected_english"])
            gt_od = _clean(gt_row["corrected_odia"])
            gemini_orig_en = _clean(gt_row["gemini_original_english"])

            if rt_rows.empty:
                rt_en, rt_od, rt_readback = "", "", ""
                notes = "no matching row in fresh Gemini run"
            else:
                rt_en = _joined_realtime_value(rt_rows, "english_value")
                rt_od = _joined_realtime_value(rt_rows, "odia_text")
                rt_readback = _joined_realtime_value(rt_rows, "latin_readback")
                notes = "; ".join(
                    n for n in rt_rows.sort_values("item_index")["notes"].tolist()
                    if _clean(n))

            # ground_truth_readback: the expert's corrected_english column, used
            # as the romanized/English-side form of their Odia correction (this
            # is what corrected_english IS in practice for place names, reg
            # numbers, etc. — see e.g. district: corrected_odia="ଅନୁଗୋଳ",
            # corrected_english="ANGUL").
            gt_readback = gt_en

            # original_metadata: the English value that was actually SENT to
            # Gemini as the grounding target for this field (ocr_value at
            # ingest). Blank here means Gemini was never given anything to
            # locate for this field — so a blank Gemini result is expected,
            # not a miss.
            original_metadata = gemini_orig_en

            en_match = _norm_compare(gt_en) == _norm_compare(rt_en) if gt_en else None
            od_match = _norm_compare(gt_od) == _norm_compare(rt_od) if gt_od else None
            readback_match = (
                _norm_compare(gt_readback) == _norm_compare(rt_readback)
                if gt_readback else None
            )

            gt_blank = (_norm_compare(gt_od) == "" and _norm_compare(gt_readback) == "")
            gemini_blank = (_norm_compare(rt_od) == "" and _norm_compare(rt_readback) == "")
            meta_blank = _norm_compare(original_metadata) in ("", ",")

            # Genuine-content verdict: ODIA-TO-ODIA ONLY. The readback columns
            # (latin_readback / corrected_english) are kept in the output purely
            # for reference — they do NOT drive the match verdict. corrected_english
            # is frequently left stale/unedited by the expert (they mainly edit
            # corrected_odia), so basing the verdict on it produces false matches
            # that mask real Odia content differences (e.g. Gemini echoing a
            # metadata serial-number suffix like "-37" the real Odia transcription
            # doesn't contain).
            gt_od_key = _content_key(gt_od, field)
            rt_od_key = _content_key(rt_od, field)
            odia_content_match = (
                gt_od_key == rt_od_key if (gt_od_key or rt_od_key) else None)

            # For place/name-type fields, also accept a tolerable spelling
            # variant or granularity difference (per the prompt's own NAMES /
            # PLACE NAMES rules) — compare each comma-split item individually
            # since these can be multi-item list fields (property/seller/buyer).
            base_f = field.split(".")[0]
            attr_f = field.split(".")[1] if "." in field else ""
            if not odia_content_match and (base_f in _FUZZY_FIELDS or attr_f in _FUZZY_ATTRS):
                gt_items = [p.strip() for p in gt_od.split(",") if p.strip()]
                rt_items = [p.strip() for p in rt_od.split(",") if p.strip()]
                if gt_items and rt_items:
                    # every Gemini item should fuzzy-match at least one GT item
                    odia_content_match = all(
                        any(_fuzzy_equivalent(gi, ri, field) for gi in gt_items)
                        for ri in rt_items
                    )
                # Cross-script fallback: sometimes the DB's "Odia" column was
                # actually typed in Latin (e.g. corrected_odia="Angul" instead
                # of "ଅନୁଗୋଳ") — a data-entry quirk, not a Gemini error. In that
                # case compare against Gemini's OWN Latin readback instead,
                # with the same fuzzy tolerance.
                if not odia_content_match and gt_od and rt_od and not _is_odia_script(gt_od):
                    rt_readback_items = [p.strip() for p in rt_readback.split(",") if p.strip()]
                    if rt_readback_items:
                        odia_content_match = all(
                            any(_fuzzy_equivalent(gi, ri, field) for gi in gt_items)
                            for ri in rt_readback_items
                        )
                # Symmetric case: Gemini itself put Latin text in odia_text
                # (a script-fidelity slip, e.g. "ANGUL" instead of "ଅନୁଗୁଳ") —
                # compare it against the expert's readback (corrected_english),
                # since rt_od is already Latin here.
                if not odia_content_match and gt_od and rt_od and _is_odia_script(gt_od) and not _is_odia_script(rt_od):
                    gt_readback_items = [p.strip() for p in gt_readback.split(",") if p.strip()]
                    if gt_readback_items:
                        odia_content_match = all(
                            any(_fuzzy_equivalent(gri, ri, field) for gri in gt_readback_items)
                            for ri in rt_items
                        )
                # Last resort: known place-name alias table (e.g. Angul
                # district variants), for when neither side has any readback
                # text to cross-check against.
                if not odia_content_match and gt_items and rt_items:
                    gt_aliases = {a for a in (_place_alias(g) for g in gt_items) if a}
                    rt_aliases = {a for a in (_place_alias(r) for r in rt_items) if a}
                    if gt_aliases and rt_aliases and gt_aliases == rt_aliases:
                        odia_content_match = True

                # VILLAGE NUMBER CHECK: a fuzzy text match on the village name
                # itself doesn't excuse a missing/different serial number
                # (e.g. GT has "-44" but Gemini's transcription doesn't, or
                # vice versa) — per the prompt's own rule, that number should
                # be transcribed whenever it's genuinely on the page, so a
                # digit mismatch here is worth surfacing even if the name text
                # otherwise reads as the same place.
                if odia_content_match and attr_f == "village" and gt_items and rt_items:
                    gt_digit_sets = [set(re.findall(r"\d+", _to_latin_digits(g))) for g in gt_items]
                    rt_digit_sets = [set(re.findall(r"\d+", _to_latin_digits(r))) for r in rt_items]
                    if any(gt_digit_sets) or any(rt_digit_sets):
                        # every Gemini item's digit set should match SOME GT
                        # item's digit set (order-agnostic, same tolerance as
                        # the name-matching above)
                        digits_ok = all(
                            any(rds == gds for gds in gt_digit_sets)
                            for rds in rt_digit_sets
                        )
                        if not digits_ok:
                            odia_content_match = False
            content_matches = odia_content_match is True

            # Classification (drives highlight colour in the viewer):
            #   match / spelling / formatting / dup -> no highlight
            #   both-blank                          -> ORANGE
            #   deed_type                           -> never red (by design)
            #   genuine content diff                -> RED
            base_field = field.split(".")[0]
            if content_matches:
                issue_type = "match"
            elif gemini_blank and (gt_blank or meta_blank):
                # Gemini returned nothing and there was nothing to find (either
                # the expert also left it blank, or no metadata anchor was sent)
                issue_type = "both-blank (nothing to locate)"
            elif base_field == "deed_type":
                # deed_type: Gemini transcribes the page's Odia term verbatim
                # while the DB stores an English category label — expected.
                issue_type = "deed_type (category vs transcription — expected)"
            else:
                issue_type = "content mismatch"

            rows.append({
                "deed_number": deed,
                "field": field,
                "original_metadata": original_metadata,
                "ground_truth_english": gt_en,
                "fresh_gemini_english": rt_en,
                "english_match": en_match,
                "ground_truth_odia": gt_od,
                "fresh_gemini_odia": rt_od,
                "odia_match": od_match,
                "ground_truth_readback": gt_readback,
                "fresh_gemini_readback": rt_readback,
                "readback_match": readback_match,
                "issue_type": issue_type,
                "gemini_original_english_at_ingest": gemini_orig_en,
                "was_corrected_by_expert": _norm_compare(gt_en) != _norm_compare(gemini_orig_en),
                "notes": notes,
            })
    return pd.DataFrame(rows)


def field_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mismatch rate per field — this is the prompt-refinement
    target list."""
    out = []
    for field, grp in comparison.groupby("field"):
        n = len(grp)
        en_checked = grp["english_match"].notna().sum()
        en_mismatches = (grp["english_match"] == False).sum()
        od_checked = grp["odia_match"].notna().sum()
        od_mismatches = (grp["odia_match"] == False).sum()
        rb_checked = grp["readback_match"].notna().sum()
        rb_mismatches = (grp["readback_match"] == False).sum()
        script_only = (grp["issue_type"] == "script-only (content correct, wrong script)").sum()
        content_mismatch = grp["issue_type"].str.startswith("content mismatch", na=False).sum()
        out.append({
            "field": field,
            "n_deeds": n,
            "english_mismatches": int(en_mismatches),
            "english_checked": int(en_checked),
            "english_accuracy_pct": round(100 * (1 - en_mismatches / en_checked), 1) if en_checked else None,
            "odia_mismatches": int(od_mismatches),
            "odia_checked": int(od_checked),
            "odia_accuracy_pct": round(100 * (1 - od_mismatches / od_checked), 1) if od_checked else None,
            "readback_mismatches": int(rb_mismatches),
            "readback_checked": int(rb_checked),
            "readback_accuracy_pct": round(100 * (1 - rb_mismatches / rb_checked), 1) if rb_checked else None,
            "script_only_issues": int(script_only),
            "genuine_content_mismatches": int(content_mismatch),
        })
    df = pd.DataFrame(out)
    return df.sort_values(["english_accuracy_pct", "odia_accuracy_pct"], na_position="first")
