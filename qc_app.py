"""
qc_app.py — tiny QC viewer: image/PDF on the left, its JSON on the right,
with prev/next navigation and a quick approve/flag button.

Each "record" = a folder containing one JSON file + one image (or PDF).
e.g.  data/R0851234/grounding.json
      data/R0851234/R0851234.pdf

LOCAL USE
    pip install streamlit pymupdf
    streamlit run qc_app.py -- --data data

GCS USE (same env vars as gcs_store.py in the deed-validator repo)
    export GCS_BUCKET=classification-vision
    export GCS_PREFIX=ocr_outputs/orissa_deeds/sample_1000
    export GCS_CREDENTIALS_JSON="$(cat key.json)"   # or leave unset to use ambient creds
    pip install streamlit pymupdf google-cloud-storage
    streamlit run qc_app.py

QC decisions (approve / flag / note) are appended to qc_results.csv next to
this script, one row per record, latest decision wins if you revisit one.
"""

import csv
import io
import json
import os
import sys
from pathlib import Path

import streamlit as st

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp")
RESULTS_CSV = Path(__file__).parent / "qc_results.csv"

st.set_page_config(page_title="QC Viewer", layout="wide")


# ---------------------------------------------------------------- sources --
class LocalSource:
    """Records read from a local folder: <root>/<record_id>/*.json + image."""

    def __init__(self, root):
        self.root = Path(root)

    def list_ids(self):
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and any(p.glob("*.json"))
        )

    def load(self, record_id):
        folder = self.root / record_id
        jpath = next(folder.glob("*.json"), None)
        data = json.loads(jpath.read_text(encoding="utf-8")) if jpath else {}

        img_path = next((p for p in folder.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS), None)
        if img_path:
            return data, ("image", img_path.read_bytes())

        pdf_path = next((p for p in folder.iterdir()
                          if p.suffix.lower() == ".pdf"), None)
        if pdf_path:
            return data, ("pdf", pdf_path.read_bytes())

        return data, (None, None)


class GcsSource:
    """Records read straight from a GCS bucket, same env vars as gcs_store.py.
    Uses ONLY storage.objects.get (never .list) — some service accounts, like
    reader accounts, only have get access, not list access on the bucket."""

    def __init__(self):
        from google.cloud import storage
        from google.oauth2 import service_account

        creds_json = os.environ.get("GCS_CREDENTIALS_JSON")
        if creds_json:
            info = json.loads(creds_json)
            creds = service_account.Credentials.from_service_account_info(info)
            client = storage.Client(credentials=creds, project=info.get("project_id"))
        else:
            client = storage.Client()
        self.bucket = client.bucket(os.environ["GCS_BUCKET"])
        self.prefix = os.environ.get("GCS_PREFIX", "").strip("/")
        self.prefix = self.prefix + "/" if self.prefix else ""

    def _blob(self, rel_path):
        return self.bucket.blob(self.prefix + rel_path)

    def _read_text(self, rel_path):
        blob = self._blob(rel_path)
        try:
            if blob.exists():
                return blob.download_as_text()
        except Exception:
            pass
        return None

    def list_ids(_self):
        """Try index.csv (needs only objects.get). If it's not there, fall
        back to whatever IDs the user pastes in manually — this account has
        no storage.objects.list permission so we can't discover them."""
        raw = _self._read_text("index.csv")
        if raw:
            import csv as _csv
            rows = list(_csv.DictReader(io.StringIO(raw)))
            ids = sorted({str(r["reg_no"]).strip() for r in rows if r.get("reg_no")})
            if ids:
                return ids
        return None  # signal "couldn't auto-discover"

    @st.cache_data(show_spinner="Loading record...")
    def load(_self, record_id):
        data = {}
        jtext = _self._read_text(f"{record_id}/grounding.json")
        if jtext:
            data = json.loads(jtext)

        for ext in IMAGE_EXTS:
            blob = _self._blob(f"{record_id}/{record_id}{ext}")
            try:
                if blob.exists():
                    return data, ("image", blob.download_as_bytes())
            except Exception:
                pass

        pdf_blob = _self._blob(f"{record_id}/{record_id}.pdf")
        try:
            if pdf_blob.exists():
                return data, ("pdf", pdf_blob.download_as_bytes())
        except Exception:
            pass

        return data, (None, None)


@st.cache_resource
def get_source(data_root):
    if os.environ.get("GCS_BUCKET"):
        return GcsSource()
    return LocalSource(data_root)


# ------------------------------------------------------------------ pdf ---
def pdf_first_page_png(pdf_bytes):
    """Render page 1 of a PDF to PNG bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")


# --------------------------------------------------------------- results --
def load_results():
    if not RESULTS_CSV.exists():
        return {}
    out = {}
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["record_id"]] = row  # later rows overwrite earlier ones
    return out


def save_result(record_id, status, note):
    results = load_results()
    results[record_id] = {"record_id": record_id, "status": status, "note": note}
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["record_id", "status", "note"])
        w.writeheader()
        for r in results.values():
            w.writerow(r)


# ------------------------------------------------------------------- ui ---
def main():
    st.title("QC Viewer")

    using_gcs = bool(os.environ.get("GCS_BUCKET"))
    data_root = st.sidebar.text_input("Local data folder", value="data",
                                       disabled=using_gcs)
    source = get_source(data_root)
    ids = source.list_ids()

    if using_gcs and ids is None:
        # No storage.objects.list permission and no index.csv found —
        # can't auto-discover record IDs, so let the user paste them in.
        st.sidebar.info(
            "Couldn't auto-discover records (no list permission / no "
            "index.csv in the bucket). Paste record IDs below, one per line."
        )
        pasted = st.sidebar.text_area("Record IDs", height=200,
                                       placeholder="910010000985\n910010000986\n...")
        ids = sorted({line.strip() for line in pasted.splitlines() if line.strip()})

    if not ids:
        st.warning(
            f"No records found under '{data_root}'. Set GCS_BUCKET / GCS_PREFIX "
            "env vars to read from GCS instead, or point this at a folder "
            "containing one subfolder per record (each with a .json + image/pdf)."
        )
        return

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = max(0, min(st.session_state.idx, len(ids) - 1))

    results = load_results()

    # ---- navigation row ----
    nav1, nav2, nav3, nav4 = st.columns([1, 1, 3, 2])
    with nav1:
        if st.button("< Prev", use_container_width=True) and st.session_state.idx > 0:
            st.session_state.idx -= 1
            st.rerun()
    with nav2:
        if st.button("Next >", use_container_width=True) and st.session_state.idx < len(ids) - 1:
            st.session_state.idx += 1
            st.rerun()
    with nav3:
        picked = st.selectbox(
            "Jump to record", ids, index=st.session_state.idx,
            label_visibility="collapsed",
        )
        if picked != ids[st.session_state.idx]:
            st.session_state.idx = ids.index(picked)
            st.rerun()
    with nav4:
        st.markdown(f"**{st.session_state.idx + 1} / {len(ids)}**")
        done = sum(1 for r in results.values() if r["status"] in ("approved", "flagged"))
        st.caption(f"{done} reviewed so far")

    record_id = ids[st.session_state.idx]
    data, (kind, raw) = source.load(record_id)

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader(record_id)
        if kind == "image":
            st.image(raw, use_container_width=True)
        elif kind == "pdf":
            png = pdf_first_page_png(raw)
            if png:
                st.image(png, use_container_width=True)
            else:
                st.info("Install `pymupdf` to preview PDFs, or view it directly:")
                st.download_button("Download PDF", raw, file_name=f"{record_id}.pdf")
        else:
            st.error("No image or PDF found for this record.")

    with right:
        st.subheader("JSON")
        search = st.text_input("Filter fields (matches key or value)", key=f"search_{record_id}")
        if search:
            fields = data.get("fields") if isinstance(data, dict) else None
            if isinstance(fields, list):
                s = search.lower()
                shown = {**{k: v for k, v in data.items() if k != "fields"},
                         "fields": [f for f in fields if s in json.dumps(f, ensure_ascii=False).lower()]}
                st.json(shown, expanded=True)
            else:
                st.json(data, expanded=True)
        else:
            st.json(data, expanded=True)

    st.divider()
    prior = results.get(record_id, {})
    a, b, c = st.columns([1, 1, 4])
    with a:
        if st.button("✅ Approve", use_container_width=True, type="primary"):
            save_result(record_id, "approved", "")
            st.session_state.idx = min(st.session_state.idx + 1, len(ids) - 1)
            st.rerun()
    with b:
        if st.button("🚩 Flag", use_container_width=True):
            save_result(record_id, "flagged", st.session_state.get(f"note_{record_id}", ""))
            st.session_state.idx = min(st.session_state.idx + 1, len(ids) - 1)
            st.rerun()
    with c:
        note = st.text_input(
            "Note (optional, saved with Flag)",
            value=prior.get("note", ""), key=f"note_{record_id}",
        )

    if prior.get("status"):
        st.caption(f"Current status: **{prior['status']}**"
                    + (f" — {prior['note']}" if prior.get("note") else ""))


if __name__ == "__main__":
    main()
