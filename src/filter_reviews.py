"""
filter_reviews.py
-----------------
Duyệt qua từng dòng trong file CSV, gọi LLM (Groq hoặc OpenAI)
để xác định dòng đó có phải là review thực sự hay không.

Các dòng KHÔNG phải review:
  - Câu hỏi xin review ("cho mình xin review", "có ai làm ở đây chưa")
  - Spam, quảng cáo, nội dung vô nghĩa
  - Bình luận không liên quan đến trải nghiệm làm việc
  - Reply ngắn không có nội dung review
  - Nội dung bị ẩn/xóa

Sử dụng:
    python filter_reviews.py reviews_cleaned_Bao.csv
    python filter_reviews.py reviews_cleaned_Bao.csv --test 10
    python filter_reviews.py reviews_cleaned_Bao.csv --output result.csv --rerun
    python filter_reviews.py reviews_cleaned_Bao.csv --model llama-3.3-70b-versatile
    python filter_reviews.py reviews_cleaned_Bao.csv --provider openai
    python filter_reviews.py reviews_cleaned_Bao.csv --provider openai --model gpt-4o-mini

Cấu trúc output CSV thêm/cập nhật cột:
    is_review : TRUE / FALSE
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
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL = "gpt-5"

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_DEFAULT_MODEL = "openai/gpt-oss-120b"

PROVIDER_CONFIG = {
    "groq": {
        "api_url": GROQ_API_URL,
        "default_model": GROQ_DEFAULT_MODEL,
        "env_key": "GROQ_API_KEY",
        "models_url": "https://api.groq.com/openai/v1/models",
    },
    "openai": {
        "api_url": OPENAI_API_URL,
        "default_model": OPENAI_DEFAULT_MODEL,
        "env_key": "OPENAI_API_KEY",
        "models_url": "https://api.openai.com/v1/models",
    },
    "together": {
        "api_url": TOGETHER_API_URL,
        "default_model": TOGETHER_DEFAULT_MODEL,
        "env_key": "TOGETHER_API_KEY",
        "models_url": "https://api.together.xyz/v1/models",
    },
}

SYSTEM_PROMPT = """\
You are an expert classifier for Vietnamese employee/company reviews.

Your task: Determine whether a given text is a GENUINE employee/candidate review or NOT.

A text is a GENUINE review (is_review = true) if:
- It describes personal experience working at or interviewing with the company
- It comments on salary, benefits, management, culture, work environment, colleagues, projects, etc.
- It can be positive, negative, or neutral — as long as it reflects a real work experience
- Even very short reviews like "công ty tệ" or "môi trường tốt" count as genuine reviews

A text is NOT a review (is_review = false) if:
- It's a question asking for reviews ("cho mình xin review", "có ai làm ở đây chưa", "mọi người review giúp")
- It's spam, advertisement, or completely unrelated to working experience
- It's nonsensical or gibberish text with no clear meaning
- It's a notice that content was hidden/removed ("nội dung đã bị ẩn")
- It's purely a comment about the company's product/service from a CUSTOMER perspective (not employee)
- It's just insults/profanity without any work-related content

Output ONLY a JSON object:
{
  "is_review": true/false,
}
"""

USER_TEMPLATE = """\
Text:
{review}
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("filter_reviews")


# ---------------------------------------------------------------------------
# LLM API call (Groq / OpenAI compatible)
# ---------------------------------------------------------------------------
def call_llm(prompt: str, model: str, api_key: str, provider: str = "groq",
             retries: int = 6, timeout: int = 120) -> str:
    cfg = PROVIDER_CONFIG[provider]
    api_url = cfg["api_url"]
    if not api_key:
        raise ValueError(f"{provider.upper()} API key chưa được cung cấp. Dùng biến môi trường {cfg['env_key']}.")
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
    # Một số model (vd: gpt-5) không hỗ trợ temperature/top_p tùy chỉnh
    if provider != "openai" or not model.startswith("gpt-5"):
        payload["temperature"] = 0.0
        payload["top_p"] = 1.0
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-requests")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = min(2 ** attempt, 60)
                logger.error("%s 429 rate-limit | attempt=%d | wait=%.1fs", provider, attempt, wait)
                if attempt <= retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"{provider} rate-limit vượt giới hạn sau {retries + 1} lần")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 * attempt, 30))
    raise RuntimeError(f"{provider} API thất bại sau {retries + 1} lần: {last_error}")


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
def parse_result(raw: str) -> dict:
    default = {
        "is_review": "ERROR",
        "reason": "",
    }
    try:
        json_str = extract_json_str(raw)
        data = json.loads(json_str)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("JSON parse error: %s | raw=%r", exc, raw[:300])
        return default

    is_review = data.get("is_review")
    if isinstance(is_review, bool):
        default["is_review"] = "TRUE" if is_review else "FALSE"
    elif isinstance(is_review, str):
        default["is_review"] = "TRUE" if is_review.lower() == "true" else "FALSE"
    else:
        default["is_review"] = "ERROR"

    default["reason"] = str(data.get("reason", "")).strip()
    return default


# ---------------------------------------------------------------------------
# Process a single row
# ---------------------------------------------------------------------------
def process_row(review_text: str, model: str, api_key: str, provider: str = "groq") -> dict:
    review = str(review_text).strip() if pd.notna(review_text) else ""

    if not review:
        return {"is_review": "FALSE", "reason": "Nội dung trống"}

    user_prompt = USER_TEMPLATE.format(review=review)
    try:
        raw = call_llm(user_prompt, model=model, api_key=api_key, provider=provider)
        return parse_result(raw)
    except Exception as exc:
        logger.error("Lỗi API | review=%r | error=%s", review[:80], exc)
        return {"is_review": "ERROR", "reason": str(exc)[:100]}


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
    parser = argparse.ArgumentParser(description="Lọc review thật / không phải review bằng LLM (Groq / OpenAI)")
    parser.add_argument("csv_file", help="Đường dẫn tới file CSV")
    parser.add_argument("--test", type=int, default=0, metavar="N",
                        help="Chỉ chạy N dòng đầu để kiểm tra")
    parser.add_argument("--rerun", action="store_true",
                        help="Xử lý lại các dòng đã có kết quả")
    parser.add_argument("--provider", default="groq", choices=["groq", "openai", "together"],
                        help="Provider LLM: groq, openai hoặc together (mặc định: groq)")
    parser.add_argument("--model", default=None,
                        help="Tên model (mặc định: tùy provider)")
    parser.add_argument("--delay", type=float, default=0.5, metavar="SEC",
                        help="Thời gian chờ (giây) giữa mỗi request (mặc định 0.5)")
    parser.add_argument("--output", default=None, metavar="CSV_PATH",
                        help="Đường dẫn file CSV output")
    parser.add_argument("--review-col", default=None,
                        help="Tên cột chứa review (tự detect nếu không chỉ định)")
    args = parser.parse_args()

    # Resolve provider config
    provider = args.provider
    cfg = PROVIDER_CONFIG[provider]
    if args.model is None:
        args.model = cfg["default_model"]

    load_dotenv()
    api_key = os.environ.get(cfg["env_key"], "")

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[LỖI] Không tìm thấy file: {csv_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else csv_path.with_name(csv_path.stem + "_filtered.csv")

    # Resume từ checkpoint nếu có
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
            "review_content_cleaned", "review_content_masked",
            "review_content", "review", "combined_review",
        ])
    if col_review is None:
        print(f"[LỖI] Không tìm thấy cột review. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)

    print(f"  → Cột review: '{col_review}'")

    # Khởi tạo cột output nếu chưa có
    if "is_review" not in df.columns:
        df["is_review"] = ""

    # Xác định dòng cần xử lý
    if args.rerun:
        mask = pd.Series([True] * len(df), index=df.index)
    else:
        mask = df["is_review"].isna() | (df["is_review"].astype(str).str.strip().isin(["", "ERROR", "None"]))

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

    print(f"File output: {output_path}")

    # Kiểm tra kết nối LLM
    print(f"\nKiểm tra kết nối {provider.upper()} (model: {args.model}) ...")
    if not api_key:
        print(f"  [LỖI] Chưa có {provider.upper()} API key. Cài đặt biến môi trường {cfg['env_key']}.")
        sys.exit(1)
    try:
        resp = requests.get(
            cfg["models_url"],
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        available = [m["id"] for m in resp.json().get("data", [])]
        if available and args.model not in available:
            print(f"  [CẢNH BÁO] Model '{args.model}' không thấy trong danh sách {provider.upper()}.")
            print(f"  Một số model có sẵn: {available[:20]}")
        else:
            print(f"  ✓ Kết nối {provider.upper()} OK.")
    except Exception as exc:
        print(f"  [CẢNH BÁO] Không kiểm tra được {provider.upper()}: {exc}")

    # Vòng lặp xử lý
    rows_to_process = df[mask].index.tolist()
    errors = 0
    true_count = 0
    false_count = 0
    SAVE_EVERY = 10

    print()
    for i, idx in enumerate(tqdm(rows_to_process, desc="Lọc review", unit="dòng")):
        result = process_row(
            review_text=df.at[idx, col_review],
            model=args.model,
            api_key=api_key,
            provider=provider,
        )

        if args.delay > 0:
            time.sleep(args.delay)

        df.at[idx, "is_review"] = result["is_review"]

        if result["is_review"] == "ERROR":
            errors += 1
        elif result["is_review"] == "TRUE":
            true_count += 1
        elif result["is_review"] == "FALSE":
            false_count += 1

        # Save checkpoint
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Lưu kết quả cuối
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu: {output_path}")
    print(f"  Tổng xử lý    : {len(rows_to_process)}")
    print(f"  Là review      : {true_count}")
    print(f"  Không phải     : {false_count}")
    print(f"  Errors         : {errors}")


if __name__ == "__main__":
    main()
