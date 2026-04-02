"""
llm_process.py
--------------
Duyệt qua từng review trong file Excel, gọi LLM (Ollama) để:
  1. Xác định có phải review công ty thật không (is_review)
  2. Mask tên công ty và các biến thể trong review (review_masked)
  3. Trích xuất các khía cạnh (aspect) và cảm xúc tương ứng (sentiment)

LLM trả về JSON → parse → ghi ra CSV.

Sử dụng:
    python llm_process.py 100_review.xlsx
    python llm_process.py 100_review.xlsx --test 5
    python llm_process.py 100_review.xlsx --output result.csv --rerun

Cấu trúc output CSV thêm các cột:
    is_review         : true/false
    review_masked     : text đã mask tên công ty
    aspect_labels     : vd "Salary & Benefits, Work Hours & Workload"
    aspect_sentiments : vd "positive, negative"
    aspects_raw       : JSON gốc từ LLM (để debug)
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

MISSING = []
try:
    import pandas as pd
except ImportError:
    MISSING.append("pandas")
try:
    import openpyxl  # noqa: F401
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
VALID_ASPECTS = [
    "Salary & Benefits",
    "Training & Learning",
    "Management & Leadership",
    "Culture & Environment",
    "Office & Workspace",
    "Work Hours & Workload",
    "Career Growth & Opportunities",
]

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

SYSTEM_PROMPT = """\
You are an expert analyzer for Vietnamese employee reviews from ITViec (a Vietnamese IT job site).
Reviews are written in Vietnamese or mixed Vietnamese-English (informal, with slang and teencode).

Your job for each review is to return a single JSON object with exactly these fields:

{
  "is_review": <true or false>,
  "review_masked": "<review text with ALL company name variants replaced by Company A B C>",
  "aspects": [
    {"aspect": "<aspect label>", "sentiment": "<positive|negative|neutral>"}
  ]
}

Rules:
1. is_review: Set to false if the text is NOT a real employee/candidate review — e.g. it's a question asking for reviews, a spam post, a nonsensical sentence, or completely unrelated to working at the company. Set to true for genuine reviews (positive or negative).

2. review_masked: Replace the company name and ALL its variants (abbreviations, partial names, etc.) with [COMPANY]. The company name will be given to you. Keep the rest of the text unchanged.

3. aspects: Extract only the aspects that are explicitly or clearly implied in the review. Each aspect must be one of:
   - Salary & Benefits
   - Training & Learning
   - Management & Leadership
   - Culture & Environment
   - Office & Workspace
   - Work Hours & Workload
   - Career Growth & Opportunities

   sentiment must be "positive", "negative", or "neutral".
   If no aspects apply, return an empty list: "aspects": []

Output ONLY the JSON object. No explanation, no markdown, no extra text.
"""

USER_TEMPLATE = """\
Company name: {company_name}

Review:
{review}
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="llm_process_errors.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------
def call_ollama(prompt: str, model: str, base_url: str, retries: int = 2, timeout: int = 180) -> str:
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
            return resp.json().get("response", "").strip()
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama API thất bại sau {retries + 1} lần: {last_error}")


# ---------------------------------------------------------------------------
# JSON extraction — strip markdown fences if LLM wraps output
# ---------------------------------------------------------------------------
def extract_json_str(raw: str) -> str:
    """Trích xuất JSON object từ raw string, bỏ qua markdown fences nếu có."""
    raw = raw.strip()
    # Strip ```json ... ``` hoặc ``` ... ```
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    # Tìm object JSON đầu tiên
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    return match.group(0) if match else raw


# ---------------------------------------------------------------------------
# Parse & validate LLM output
# ---------------------------------------------------------------------------
def parse_result(raw: str, fallback_text: str) -> dict:
    """
    Parse JSON từ LLM output.
    Trả về dict với các key: is_review, review_masked, aspects_raw,
    aspect_labels, aspect_sentiments.
    """
    default = {
        "is_review": None,
        "review_masked": fallback_text,
        "aspects_raw": "[]",
        "aspect_labels": "",
        "aspect_sentiments": "",
    }

    try:
        json_str = extract_json_str(raw)
        data = json.loads(json_str)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("JSON parse error: %s | raw=%r", exc, raw[:200])
        default["aspects_raw"] = "PARSE_ERROR"
        return default

    # is_review
    is_review = data.get("is_review")
    if isinstance(is_review, str):
        is_review = is_review.lower() == "true"
    default["is_review"] = bool(is_review) if is_review is not None else None

    # review_masked
    masked = data.get("review_masked", "")
    default["review_masked"] = str(masked).strip() if masked else fallback_text

    # aspects
    aspects = data.get("aspects", [])
    if not isinstance(aspects, list):
        aspects = []

    valid_aspects = []
    for item in aspects:
        if not isinstance(item, dict):
            continue
        aspect = str(item.get("aspect", "")).strip()
        sentiment = str(item.get("sentiment", "")).strip().lower()
        # Fuzzy match aspect label
        matched = next((a for a in VALID_ASPECTS if a.lower() == aspect.lower()), None)
        if matched is None:
            # Partial match
            matched = next((a for a in VALID_ASPECTS if aspect.lower() in a.lower() or a.lower() in aspect.lower()), None)
        if matched and sentiment in VALID_SENTIMENTS:
            valid_aspects.append({"aspect": matched, "sentiment": sentiment})

    default["aspects_raw"] = json.dumps(valid_aspects, ensure_ascii=False)
    default["aspect_labels"] = ", ".join(a["aspect"] for a in valid_aspects)
    default["aspect_sentiments"] = ", ".join(a["sentiment"] for a in valid_aspects)

    return default


# ---------------------------------------------------------------------------
# Process a single row
# ---------------------------------------------------------------------------
def process_row(review_text: str, company_name: str, model: str, base_url: str) -> dict:
    review = str(review_text).strip() if pd.notna(review_text) else ""
    company = str(company_name).strip() if pd.notna(company_name) else ""

    if not review:
        return {
            "is_review": False,
            "review_masked": "",
            "aspects_raw": "[]",
            "aspect_labels": "",
            "aspect_sentiments": "",
        }

    prompt = USER_TEMPLATE.format(company_name=company, review=review)
    try:
        raw = call_ollama(prompt, model=model, base_url=base_url)
        return parse_result(raw, fallback_text=review)
    except Exception as exc:
        logger.error("Lỗi API | company=%r | review=%r | error=%s", company, review[:80], exc)
        return {
            "is_review": None,
            "review_masked": review,
            "aspects_raw": "ERROR",
            "aspect_labels": "ERROR",
            "aspect_sentiments": "ERROR",
        }


# ---------------------------------------------------------------------------
# Column resolver
# ---------------------------------------------------------------------------
def find_col(df: pd.DataFrame, candidates: list) -> str | None:
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
    parser = argparse.ArgumentParser(description="Xử lý review bằng LLM: lọc, mask, trích aspect")
    parser.add_argument("excel_file", help="Đường dẫn tới file Excel (.xlsx)")
    parser.add_argument("--test", type=int, default=0, metavar="N",
                        help="Chỉ chạy N dòng đầu để kiểm tra")
    parser.add_argument("--rerun", action="store_true",
                        help="Xử lý lại các dòng đã có kết quả")
    parser.add_argument("--model", default="gpt-oss:20b",
                        help="Tên model Ollama (mặc định: gpt-oss:20b)")
    parser.add_argument("--ollama-url", default="http://14.224.236.84:8003/",
                        help="URL Ollama server")
    parser.add_argument("--sheet", default=0,
                        help="Tên hoặc index sheet Excel")
    parser.add_argument("--output", default=None, metavar="CSV_PATH",
                        help="Đường dẫn file CSV output")
    args = parser.parse_args()

    excel_path = Path(args.excel_file)
    if not excel_path.exists():
        print(f"[LỖI] Không tìm thấy file: {excel_path}")
        sys.exit(1)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    print(f"Đọc file: {excel_path}  (sheet: {sheet})")
    df = pd.read_excel(excel_path, sheet_name=sheet, dtype=str)
    print(f"  → {len(df)} dòng, {len(df.columns)} cột: {list(df.columns)}")

    # Tìm cột review và company
    col_review = find_col(df, ["review_content_masked", "review_content", "review", "combined_review"])
    col_company = find_col(df, ["company_name", "company"])

    if col_review is None:
        print(f"[LỖI] Không tìm thấy cột review. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)
    if col_company is None:
        print(f"[LỖI] Không tìm thấy cột company_name. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)

    print(f"  → Cột review  : '{col_review}'")
    print(f"  → Cột company : '{col_company}'")

    # Khởi tạo cột output nếu chưa có
    for col in ["is_review", "review_masked", "aspect_labels", "aspect_sentiments", "aspects_raw"]:
        if col not in df.columns:
            df[col] = ""

    # Xác định dòng cần xử lý
    if args.rerun:
        mask = pd.Series([True] * len(df))
    else:
        mask = df["is_review"].isna() | (df["is_review"].astype(str).str.strip() == "")

    if args.test > 0:
        test_indices = df[mask].head(args.test).index
        mask = pd.Series(False, index=df.index)
        mask[test_indices] = True
        print(f"\n[TEST MODE] Chỉ xử lý {mask.sum()} dòng.\n")
    else:
        print(f"\nSẽ xử lý {mask.sum()} / {len(df)} dòng chưa có kết quả.\n")

    if mask.sum() == 0:
        print("Tất cả các dòng đã được xử lý. Dùng --rerun để chạy lại.")
        sys.exit(0)

    # Output path
    output_path = Path(args.output) if args.output else excel_path.with_name(excel_path.stem + "_processed.csv")
    print(f"File output: {output_path}")

    # Kiểm tra kết nối Ollama
    print(f"\nKiểm tra kết nối Ollama tại {args.ollama_url} ...")
    try:
        resp = requests.get(args.ollama_url.rstrip("/") + "/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if models and args.model not in models:
            print(f"  [CẢNH BÁO] Model '{args.model}' không thấy trong: {models}")
        else:
            print(f"  ✓ Kết nối OK.")
    except Exception as exc:
        print(f"  [CẢNH BÁO] Không kiểm tra được Ollama: {exc}")

    # Vòng lặp xử lý
    rows_to_process = df[mask].index.tolist()
    errors = 0
    skipped_not_review = 0

    print()
    for idx in tqdm(rows_to_process, desc="Xử lý", unit="dòng"):
        result = process_row(
            review_text=df.at[idx, col_review],
            company_name=df.at[idx, col_company],
            model=args.model,
            base_url=args.ollama_url,
        )
        df.at[idx, "is_review"] = str(result["is_review"]) if result["is_review"] is not None else ""
        df.at[idx, "review_masked"] = result["review_masked"]
        df.at[idx, "aspect_labels"] = result["aspect_labels"]
        df.at[idx, "aspect_sentiments"] = result["aspect_sentiments"]
        df.at[idx, "aspects_raw"] = result["aspects_raw"]

        if result["aspect_labels"] == "ERROR":
            errors += 1
        if result["is_review"] is False:
            skipped_not_review += 1

    # Lưu kết quả
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu: {output_path}")
    print(f"  Tổng dòng xử lý   : {len(rows_to_process)}")
    print(f"  Không phải review  : {skipped_not_review}")
    print(f"  Lỗi API            : {errors}")
    if errors:
        print(f"  Chi tiết lỗi       : llm_process_errors.log")

    # Thống kê aspect
    print("\n--- Thống kê Aspect ---")
    aspect_counts: dict[str, int] = {a: 0 for a in VALID_ASPECTS}
    for val in df.loc[mask, "aspect_labels"].dropna():
        for part in str(val).split(","):
            part = part.strip()
            if part in aspect_counts:
                aspect_counts[part] += 1

    for aspect, cnt in sorted(aspect_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"  {aspect:<40} {cnt}")


if __name__ == "__main__":
    main()
