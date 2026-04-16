"""
check_mask.py
-------------
Duyệt qua từng review trong file CSV, gọi LLM (Groq) để kiểm tra xem
các ký tự mask (***) đã đúng hay chưa.
  - Nếu đúng → giữ nguyên.
  - Nếu sai  → khôi phục lại nội dung gốc.

Sử dụng:
    python check_mask.py reviews_mask.csv
    python check_mask.py reviews_mask.csv --test 5
    python check_mask.py reviews_mask.csv --output result.csv --rerun
    python check_mask.py reviews_mask.csv --model oss-120b

Cấu trúc output CSV thêm các cột:
    mask_status       : Correct / Incorrect
    recovered_content : Nội dung review sau khi khôi phục (nếu sai)
"""

import argparse
import json
import logging
import os
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
    import requests
except ImportError:
    MISSING.append("requests")
try:
    from tqdm import tqdm
except ImportError:
    MISSING.append("tqdm")
try:
    from dotenv import load_dotenv
except ImportError:
    MISSING.append("python-dotenv")

if MISSING:
    print("Thiếu các thư viện sau. Hãy cài bằng lệnh:")
    print(f"  pip install {' '.join(MISSING)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "oss-120b"

SYSTEM_PROMPT = """\
You are an expert in Natural Language Processing (NLP) with an in-depth understanding of the IT labor market in Vietnam.

Context: I have a dataset of company reviews from ITviec. In this dataset, proper names (individuals, projects) and company names are often masked using asterisks (*). However, there are two specific scenarios:

Correct Masking: The number of asterisks matches the character count of the hidden word (e.g., 'Sếp Tùng' -> 'Sếp T***').

Incorrect Masking: The asterisks are placed inconsistently, do not match the character length, or accidentally mask common words instead of proper names.

Your Task:
Analyze the review content provided below and perform the following steps:

Determine whether the asterisk-containing phrases (*) in the text are masked logically and are easy to interpret (Classify as: 'Correct' or 'Incorrect').

If 'Incorrect' (faulty masking or loss of crucial information): Use the context of the entire review to recover or restore the most likely word/name.

If 'Correct': Retain the original masked text as it is.

Return the final result strictly in JSON format.

Output Format Requirement (Return ONLY JSON):
{
  "status": "Correct/Incorrect",
  "recovered_content": "The entire review content after recovery or normalization of the masks"
}
"""

USER_TEMPLATE = """\
Review Data:
'{review}'
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="check_mask_errors.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------
def call_groq(prompt: str, model: str, api_key: str, retries: int = 6, timeout: int = 120) -> str:
    if not api_key:
        raise ValueError("Groq API key chưa được cung cấp. Dùng biến môi trường GROQ_API_KEY.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-requests")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = min(2 ** attempt, 60)
                logger.error("Groq 429 rate-limit | attempt=%d | wait=%.1fs", attempt, wait)
                if attempt <= retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Groq rate-limit vượt giới hạn sau {retries + 1} lần")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 * attempt, 30))
    raise RuntimeError(f"Groq API thất bại sau {retries + 1} lần: {last_error}")


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def extract_json_str(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    return match.group(0) if match else raw


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------
def parse_result(raw: str, fallback_text: str) -> dict:
    default = {
        "mask_status": "",
        "recovered_content": fallback_text,
    }
    try:
        json_str = extract_json_str(raw)
        data = json.loads(json_str)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("JSON parse error: %s | raw=%r", exc, raw[:300])
        default["mask_status"] = "PARSE_ERROR"
        return default

    status = str(data.get("status", "")).strip()
    if status.lower() in ("correct", "incorrect"):
        default["mask_status"] = status.capitalize()
    else:
        default["mask_status"] = status or "PARSE_ERROR"

    recovered = data.get("recovered_content", "")
    default["recovered_content"] = str(recovered).strip() if recovered else fallback_text

    return default


# ---------------------------------------------------------------------------
# Process a single row
# ---------------------------------------------------------------------------
def process_row(review_text: str, model: str, api_key: str) -> dict:
    review = str(review_text).strip() if pd.notna(review_text) else ""

    if not review or "*" not in review:
        return {
            "mask_status": "No_Mask",
            "recovered_content": review,
        }

    user_prompt = USER_TEMPLATE.format(review=review)
    try:
        raw = call_groq(user_prompt, model=model, api_key=api_key)
        return parse_result(raw, fallback_text=review)
    except Exception as exc:
        logger.error("Lỗi API | review=%r | error=%s", review[:80], exc)
        return {
            "mask_status": "ERROR",
            "recovered_content": review,
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
    parser = argparse.ArgumentParser(description="Kiểm tra mask *** trong review và khôi phục nếu sai")
    parser.add_argument("csv_file", help="Đường dẫn tới file CSV")
    parser.add_argument("--test", type=int, default=0, metavar="N",
                        help="Chỉ chạy N dòng đầu để kiểm tra")
    parser.add_argument("--rerun", action="store_true",
                        help="Xử lý lại các dòng đã có kết quả")
    parser.add_argument("--model", default=GROQ_DEFAULT_MODEL,
                        help=f"Tên model Groq (mặc định: {GROQ_DEFAULT_MODEL})")
    parser.add_argument("--delay", type=float, default=0.5, metavar="SEC",
                        help="Thời gian chờ (giây) giữa mỗi request (mặc định 0.5)")
    parser.add_argument("--output", default=None, metavar="CSV_PATH",
                        help="Đường dẫn file CSV output")
    parser.add_argument("--review-col", default=None,
                        help="Tên cột chứa review cần kiểm tra (tự detect nếu không chỉ định)")
    args = parser.parse_args()

    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[LỖI] Không tìm thấy file: {csv_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else csv_path.with_name(csv_path.stem + "_checked.csv")

    if not args.rerun and output_path.exists():
        print(f"Tìm thấy file checkpoint, tiếp tục từ: {output_path}")
        df = pd.read_csv(output_path, dtype=str, encoding="utf-8-sig")
    else:
        print(f"Đọc file: {csv_path}")
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    print(f"  → {len(df)} dòng, {len(df.columns)} cột: {list(df.columns)}")

    # Tìm cột review
    if args.review_col:
        col_review = args.review_col
        if col_review not in df.columns:
            print(f"[LỖI] Cột '{col_review}' không tồn tại. Các cột: {list(df.columns)}")
            sys.exit(1)
    else:
        col_review = find_col(df, [
            "review_content_masked", "review_masked", "review_content_cleaned",
            "review_content", "review", "combined_review",
        ])
    if col_review is None:
        print(f"[LỖI] Không tìm thấy cột review. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)

    print(f"  → Cột review: '{col_review}'")

    # Khởi tạo cột output nếu chưa có
    for col in ["mask_status", "recovered_content"]:
        if col not in df.columns:
            df[col] = ""

    # Xác định dòng cần xử lý: chỉ xử lý dòng có chứa *
    has_mask = df[col_review].fillna("").str.contains(r"\*", regex=True)

    if args.rerun:
        mask = has_mask.copy()
    else:
        not_done = df["mask_status"].isna() | (df["mask_status"].astype(str).str.strip().isin(["", "ERROR", "PARSE_ERROR"]))
        mask = has_mask & not_done

    # Đánh dấu các dòng không có * là No_Mask
    no_mask_rows = ~has_mask & (df["mask_status"].isna() | (df["mask_status"].astype(str).str.strip() == ""))
    df.loc[no_mask_rows, "mask_status"] = "No_Mask"
    df.loc[no_mask_rows, "recovered_content"] = df.loc[no_mask_rows, col_review]

    if args.test > 0:
        test_indices = df[mask].head(args.test).index
        mask = pd.Series(False, index=df.index)
        mask[test_indices] = True
        print(f"\n[TEST MODE] Chỉ xử lý {mask.sum()} dòng.\n")
    else:
        print(f"\nSẽ xử lý {mask.sum()} / {len(df)} dòng có mask cần kiểm tra.\n")

    if mask.sum() == 0:
        print("Không có dòng nào cần xử lý. Dùng --rerun để chạy lại.")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✓ Đã lưu: {output_path}")
        sys.exit(0)

    print(f"File output: {output_path}")

    # Kiểm tra kết nối Groq
    print(f"\nKiểm tra kết nối Groq (model: {args.model}) ...")
    if not groq_api_key:
        print("  [LỖI] Chưa có Groq API key. Cài đặt biến môi trường GROQ_API_KEY.")
        sys.exit(1)
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {groq_api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        available = [m["id"] for m in resp.json().get("data", [])]
        if available and args.model not in available:
            print(f"  [CẢNH BÁO] Model '{args.model}' không thấy trong danh sách Groq.")
            print(f"  Các model có sẵn: {available}")
        else:
            print("  ✓ Kết nối Groq OK.")
    except Exception as exc:
        print(f"  [CẢNH BÁO] Không kiểm tra được Groq: {exc}")

    # Vòng lặp xử lý
    rows_to_process = df[mask].index.tolist()
    errors = 0
    correct_count = 0
    incorrect_count = 0
    SAVE_EVERY = 10

    print()
    for i, idx in enumerate(tqdm(rows_to_process, desc="Kiểm tra mask", unit="dòng")):
        result = process_row(
            review_text=df.at[idx, col_review],
            model=args.model,
            api_key=groq_api_key,
        )

        if args.delay > 0:
            time.sleep(args.delay)

        df.at[idx, "mask_status"] = result["mask_status"]
        df.at[idx, "recovered_content"] = result["recovered_content"]

        if result["mask_status"] == "ERROR":
            errors += 1
        elif result["mask_status"] == "Correct":
            correct_count += 1
        elif result["mask_status"] == "Incorrect":
            incorrect_count += 1

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Lưu kết quả cuối
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu: {output_path}")
    print(f"  Tổng xử lý : {len(rows_to_process)}")
    print(f"  Correct     : {correct_count}")
    print(f"  Incorrect   : {incorrect_count}")
    print(f"  Errors      : {errors}")


if __name__ == "__main__":
    main()
