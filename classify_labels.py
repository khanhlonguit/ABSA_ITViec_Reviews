"""
classify_labels.py
------------------
Đọc file Excel với cột 'Combined_Review', phân loại từng dòng theo 5 nhãn bằng Ollama,
ghi kết quả ra file CSV mới (tên gốc + '_labeled.csv'). File Excel gốc không bị thay đổi.

Sử dụng:
    python classify_labels.py <path_to_excel> [--test N] [--rerun] [--ollama-url URL] [--output PATH]

Ví dụ:
    python classify_labels.py Vietnamese_100_cleaned.xlsx
    python classify_labels.py Vietnamese_100_cleaned.xlsx --test 5
    python classify_labels.py Vietnamese_100_cleaned.xlsx --output result.csv
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
MISSING = []
try:
    import pandas as pd
except ImportError:
    MISSING.append("pandas")
try:
    import openpyxl  # noqa: F401 – needed by pandas for Excel I/O
except ImportError:
    MISSING.append("openpyxl")
try:
    import requests
except ImportError:
    MISSING.append("requests")
try:
    from tqdm import tqdm
except ImportError:
    MISSING.append("tqdm")

if MISSING:
    print("Thiếu các thư viện sau. Hãy cài bằng lệnh:")
    print(f"  pip install {' '.join(MISSING)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_LABELS = [
    "Salary & benefits",
    "Training & learning",
    "Management cares about me",
    "Culture & fun",
    "Office & workspace",
]

SYSTEM_PROMPT = """\
You are a classifier that assigns employee-review text to one or more of the following labels:
1. Salary & benefits
2. Training & learning
3. Management cares about me
4. Culture & fun
5. Office & workspace

Rules:
- Return ONLY the matching label names, separated by commas.
- Use the exact label names as listed above (case-sensitive).
- Do NOT include explanations, numbering, or any other text.
- If none of the labels fit, return: None
"""

USER_TEMPLATE = """\
Review: {review}

Which labels apply? (comma-separated, exact names only)"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="classify_errors.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------
def call_ollama(prompt: str, model: str, base_url: str, retries: int = 2, timeout: int = 120) -> str:
    """Gọi Ollama generate API, trả về text response."""
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama API thất bại sau {retries + 1} lần: {last_error}")


# ---------------------------------------------------------------------------
# Parse & validate labels from LLM output
# ---------------------------------------------------------------------------
def parse_labels(raw: str) -> str:
    """
    Lọc chỉ giữ lại các nhãn hợp lệ từ output của LLM.
    Trả về chuỗi nhãn phân cách bằng ', ' hoặc 'N/A'.
    """
    if not raw or raw.strip().lower() in ("none", "n/a", ""):
        return "N/A"

    found = []
    for label in VALID_LABELS:
        # Tìm nhãn (không phân biệt hoa thường) trong output
        if re.search(re.escape(label), raw, flags=re.IGNORECASE):
            found.append(label)

    return ", ".join(found) if found else "N/A"


# ---------------------------------------------------------------------------
# Classify a single row
# ---------------------------------------------------------------------------
def classify_row(review_text, model: str, base_url: str) -> str:
    def safe(val) -> str:
        return str(val).strip() if pd.notna(val) and str(val).strip() else ""

    review = safe(review_text)

    if not review:
        return "N/A"

    prompt = USER_TEMPLATE.format(review=review)
    try:
        raw = call_ollama(prompt, model=model, base_url=base_url)
        return parse_labels(raw)
    except Exception as exc:
        logger.error("Lỗi khi phân loại | review=%r | error=%s", review[:100], exc)
        return "ERROR"


# ---------------------------------------------------------------------------
# Column name resolver (case-insensitive, strip whitespace)
# ---------------------------------------------------------------------------
def find_col(df: "pd.DataFrame", candidates: list[str]) -> str | None:
    normalized = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        match = normalized.get(cand.strip().lower())
        if match is not None:
            return match
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phân loại nhãn từ file Excel bằng Ollama")
    parser.add_argument("excel_file", help="Đường dẫn tới file Excel (.xlsx)")
    parser.add_argument("--test", type=int, default=0, metavar="N",
                        help="Chỉ chạy N dòng đầu để kiểm tra (mặc định: chạy toàn bộ)")
    parser.add_argument("--rerun", action="store_true",
                        help="Phân loại lại các dòng đã có nhãn rồi")
    parser.add_argument("--model", default="gpt-oss:20b",
                        help="Tên model Ollama (mặc định: gpt-oss:20b)")
    parser.add_argument("--ollama-url", default="http://113.161.88.63:8003",
                        help="URL của Ollama server (mặc định: http://113.161.88.63:8003)")
    parser.add_argument("--sheet", default=0,
                        help="Tên hoặc index sheet (mặc định: sheet đầu tiên)")
    parser.add_argument("--output", default=None, metavar="CSV_PATH",
                        help="Đường dẫn file CSV output (mặc định: <tên_file>_labeled.csv)")
    args = parser.parse_args()

    excel_path = Path(args.excel_file)
    if not excel_path.exists():
        print(f"[LỖI] Không tìm thấy file: {excel_path}")
        sys.exit(1)

    # ── Đọc Excel ──────────────────────────────────────────────────────────
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    print(f"Đọc file: {excel_path}  (sheet: {sheet})")
    df = pd.read_excel(excel_path, sheet_name=sheet, dtype=str)
    print(f"  → {len(df)} dòng, {len(df.columns)} cột")
    print(f"  → Các cột: {list(df.columns)}")

    # ── Tìm cột đầu vào ────────────────────────────────────────────────────
    col_review = find_col(df, ["Combined_Review", "combined_review", "review", "Review"])

    if col_review is None:
        print(f"[LỖI] Không tìm thấy cột 'Combined_Review' hoặc tương tự")
        print(f"       Các cột hiện có: {list(df.columns)}")
        sys.exit(1)

    print(f"\nCột đầu vào xác định:")
    print(f"  Combined_Review → '{col_review}'")

    # ── Cột Labels ─────────────────────────────────────────────────────────
    if "Labels" not in df.columns:
        df["Labels"] = ""

    # ── Chọn dòng cần xử lý ────────────────────────────────────────────────
    if args.rerun:
        mask = pd.Series([True] * len(df))
    else:
        mask = df["Labels"].isna() | (df["Labels"].astype(str).str.strip() == "")

    if args.test > 0:
        # Lấy tối đa N dòng đầu trong mask
        test_indices = df[mask].head(args.test).index
        mask = pd.Series(False, index=df.index)
        mask[test_indices] = True
        print(f"\n[TEST MODE] Chỉ xử lý {mask.sum()} dòng đầu tiên chưa có nhãn.\n")
    else:
        print(f"\nSẽ xử lý {mask.sum()} / {len(df)} dòng (chưa có nhãn).\n")

    if mask.sum() == 0:
        print("Tất cả các dòng đã có nhãn. Dùng --rerun để phân loại lại.")
        sys.exit(0)

    # ── Xác định đường dẫn CSV output ──────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = excel_path.with_name(excel_path.stem + "_labeled.csv")
    print(f"File output CSV sẽ được tạo tại: {output_path}")

    # ── Kiểm tra kết nối Ollama ────────────────────────────────────────────
    print(f"Kiểm tra kết nối Ollama tại {args.ollama_url} ...")
    try:
        resp = requests.get(args.ollama_url.rstrip("/") + "/api/tags", timeout=10)
        resp.raise_for_status()
        models_available = [m["name"] for m in resp.json().get("models", [])]
        if args.model not in models_available and models_available:
            print(f"  [CẢNH BÁO] Model '{args.model}' không thấy trong danh sách: {models_available}")
        else:
            print(f"  ✓ Kết nối OK. Model '{args.model}' sẵn sàng.")
    except Exception as exc:
        print(f"  [CẢNH BÁO] Không kiểm tra được Ollama: {exc}")
        print("  Script vẫn tiếp tục chạy...")

    # ── Phân loại ──────────────────────────────────────────────────────────
    rows_to_process = df[mask].index.tolist()
    errors = 0

    for idx in tqdm(rows_to_process, desc="Phân loại", unit="dòng"):
        label = classify_row(
            review_text=df.at[idx, col_review],
            model=args.model,
            base_url=args.ollama_url,
        )
        df.at[idx, "Labels"] = label
        if label == "ERROR":
            errors += 1

    # ── Lưu kết quả ra CSV mới ─────────────────────────────────────────────
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã tạo file CSV: {output_path}")
    print(f"  Tổng dòng xử lý  : {len(rows_to_process)}")
    print(f"  Lỗi              : {errors}")
    if errors:
        print(f"  Chi tiết lỗi     : classify_errors.log")

    # ── Thống kê nhãn ──────────────────────────────────────────────────────
    print("\n--- Thống kê nhãn ---")
    label_counts: dict[str, int] = {lbl: 0 for lbl in VALID_LABELS}
    label_counts["N/A"] = 0
    label_counts["ERROR"] = 0

    for val in df.loc[mask, "Labels"].dropna():
        for part in str(val).split(","):
            part = part.strip()
            if part in label_counts:
                label_counts[part] += 1
            elif part not in label_counts:
                label_counts[part] = 1

    for lbl, cnt in label_counts.items():
        if cnt > 0:
            print(f"  {lbl:<35} {cnt}")


if __name__ == "__main__":
    main()
