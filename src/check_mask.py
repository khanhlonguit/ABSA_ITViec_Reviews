"""
check_mask.py
-------------
Duyệt qua từng review trong file CSV, gọi LLM (ChatGPT Plus local proxy) để kiểm tra xem
các ký tự mask (***) đã đúng hay chưa.
  - Nếu đúng → giữ nguyên.
  - Nếu sai  → khôi phục lại nội dung gốc.

Sử dụng:
    python src/check_mask.py reviews_mask.csv
    python src/check_mask.py reviews_mask.csv --test 5
    python src/check_mask.py reviews_mask.csv --output result.csv --rerun
    python src/check_mask.py reviews_mask.csv --model gpt-5.2

Cấu trúc output CSV thêm các cột:
    is_masked            : true / false / ERROR / PARSE_ERROR
    review_content_masked: Nội dung review sau khi đã mask tên người/công ty
    masked_details       : Mô tả các thực thể đã mask (vd: 'FPT → [COMPANY]')
"""

import argparse
import json
import logging
import os
import re
from log_setup import setup_logging
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
CHATGPT_PLUS_API_URL = "http://localhost:8317/v1/chat/completions"
CHATGPT_PLUS_DEFAULT_MODEL = "gpt-5.2"
CHATGPT_PLUS_DEFAULT_API_KEY = "proxypal-local"

SYSTEM_PROMPT = """\
Role: You are a Data Privacy Expert and NLP Specialist. Your task is to anonymize Vietnamese company reviews by masking sensitive entities.

Task:

1. Identify all mentions of Company Names (including the specific company provided below, its abbreviations, or nicknames).
2. Identify all mentions of Person Names (including managers, CEOs, staff, or colleagues).
3. Determine if the review contains any unmasked sensitive entities.
4. If masking is needed, replace Person Names with [PERSON] and Company Names with [COMPANY].

Guidelines:

 - Do not mask technical terms, programming languages, or general positions (e.g., "Sếp", "Developer", "HR") unless they are followed by a specific name.
 - If the review is already perfectly masked or contains no names, set is_masked to false

Output Format Requirement (Return ONLY JSON):
{
  "is_masked": <true if any name/company was identified and masked, false otherwise>,
  "review_content_masked": "<full review with [PERSON]/[COMPANY] tags — ONLY if is_masked is true, otherwise empty string>",
  "masked_details": "<e.g. 'FPT Software → [COMPANY], Anh Tùng → [PERSON]' — ONLY if is_masked is true, otherwise empty string>"
}
"""

USER_TEMPLATE = """\
Review Data:
'{review}'
company_name:
'{company}'
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("check_mask")


# ---------------------------------------------------------------------------
# ChatGPT Plus local proxy API
# ---------------------------------------------------------------------------
def call_chatgpt_plus(prompt: str, model: str, api_key: str, retries: int = 4, timeout: int = 120) -> str:
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
        "stream": False,
    }
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(CHATGPT_PLUS_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
                logger.error("ChatGPT Plus 429 rate-limit | attempt=%d | wait=%.1fs", attempt, wait)
                if attempt <= retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"ChatGPT Plus rate-limit vượt giới hạn sau {retries + 1} lần")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 * attempt, 30))
    raise RuntimeError(f"ChatGPT Plus API thất bại sau {retries + 1} lần: {last_error}")


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
    try:
        json_str = extract_json_str(raw)
        data = json.loads(json_str)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("JSON parse error: %s | raw=%r", exc, raw[:300])
        return {"is_masked": "PARSE_ERROR", "review_content_masked": "", "masked_details": ""}

    is_masked = data.get("is_masked", False)
    if isinstance(is_masked, str):
        is_masked = is_masked.lower() == "true"

    if not is_masked:
        return {"is_masked": False, "review_content_masked": "", "masked_details": ""}

    masked_text = data.get("review_content_masked")
    masked_details = data.get("masked_details")
    return {
        "is_masked": True,
        "review_content_masked": str(masked_text).strip() if masked_text else "",
        "masked_details": str(masked_details).strip() if masked_details else "",
    }


# ---------------------------------------------------------------------------
# Process a single row
# ---------------------------------------------------------------------------
def process_row(
    review_text: str,
    company_name: str,
    model: str,
    chatgpt_plus_api_key: str = CHATGPT_PLUS_DEFAULT_API_KEY,
) -> dict:
    review = str(review_text).strip() if pd.notna(review_text) else ""
    company = str(company_name).strip() if pd.notna(company_name) else ""

    if not review:
        return {"is_masked": False, "review_content_masked": "", "masked_details": ""}

    user_prompt = USER_TEMPLATE.format(review=review, company=company)
    try:
        raw = call_chatgpt_plus(user_prompt, model=model, api_key=chatgpt_plus_api_key)
        return parse_result(raw, fallback_text=review)
    except Exception as exc:
        logger.error("Lỗi API | company=%r | review=%r | error=%s", company, review[:80], exc)
        return {"is_masked": "ERROR", "review_content_masked": "", "masked_details": ""}


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
    parser.add_argument("--model", default=CHATGPT_PLUS_DEFAULT_MODEL,
                        help=f"Tên model ChatGPT Plus proxy (mặc định: {CHATGPT_PLUS_DEFAULT_MODEL})")
    parser.add_argument("--delay", type=float, default=0.5, metavar="SEC",
                        help="Thời gian chờ (giây) giữa mỗi request (mặc định 0.5)")
    parser.add_argument("--output", default=None, metavar="CSV_PATH",
                        help="Đường dẫn file CSV output")
    parser.add_argument("--review-col", default=None,
                        help="Tên cột chứa review cần kiểm tra (tự detect nếu không chỉ định)")
    args = parser.parse_args()

    load_dotenv()
    chatgpt_plus_api_key = os.environ.get("CHATGPT_PLUS_API_KEY", CHATGPT_PLUS_DEFAULT_API_KEY)

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
            "review_content_corrected", "review_content_masked", "review_masked",
            "review_content_cleaned", "review_content", "review", "combined_review",
        ])
    if col_review is None:
        print(f"[LỖI] Không tìm thấy cột review. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)

    print(f"  → Cột review: '{col_review}'")

    col_company = find_col(df, ["company_name", "company"])
    if col_company is None:
        print("  [CẢNH BÁO] Không tìm thấy cột company_name, sẽ để trống.")
    else:
        print(f"  → Cột company: '{col_company}'")

    # Khởi tạo cột output nếu chưa có
    for col in ["is_masked", "review_content_masked", "masked_details"]:
        if col not in df.columns:
            df[col] = ""

    # Xác định dòng cần xử lý
    has_text = df[col_review].fillna("").str.strip() != ""

    if args.rerun:
        mask = has_text.copy()
    else:
        done_mask = df["is_masked"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])
        mask = has_text & ~done_mask

    if args.test > 0:
        test_indices = df[mask].head(args.test).index
        mask = pd.Series(False, index=df.index)
        mask[test_indices] = True
        print(f"\n[TEST MODE] Chỉ xử lý {mask.sum()} dòng.\n")
    else:
        print(f"\nSẽ xử lý {mask.sum()} / {len(df)} dòng cần mask.\n")

    if mask.sum() == 0:
        print("Không có dòng nào cần xử lý. Dùng --rerun để chạy lại.")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✓ Đã lưu: {output_path}")
        sys.exit(0)

    print(f"File output: {output_path}")
    print(f"\nDùng ChatGPT Plus local proxy tại {CHATGPT_PLUS_API_URL} (model: {args.model}) ...")

    # Vòng lặp xử lý
    rows_to_process = df[mask].index.tolist()
    errors = 0
    masked_count = 0
    SAVE_EVERY = 10

    print()
    for i, idx in enumerate(tqdm(rows_to_process, desc="Mask review", unit="dòng")):
        result = process_row(
            review_text=df.at[idx, col_review],
            company_name=df.at[idx, col_company] if col_company else "",
            model=args.model,
            chatgpt_plus_api_key=chatgpt_plus_api_key,
        )

        if args.delay > 0:
            time.sleep(args.delay)

        df.at[idx, "is_masked"] = str(result["is_masked"])
        df.at[idx, "review_content_masked"] = result["review_content_masked"]
        df.at[idx, "masked_details"] = result["masked_details"]

        if str(result["is_masked"]).upper() == "ERROR":
            errors += 1
        elif result["is_masked"] is True:
            masked_count += 1

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Lưu kết quả cuối
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu: {output_path}")
    print(f"  Tổng xử lý : {len(rows_to_process)}")
    print(f"  Có mask     : {masked_count}")
    print(f"  Không mask  : {len(rows_to_process) - masked_count - errors}")
    print(f"  Lỗi API    : {errors}")
    if errors:
        print(f"  Chi tiết lỗi: logs/")


if __name__ == "__main__":
    main()
