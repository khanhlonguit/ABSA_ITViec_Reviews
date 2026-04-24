"""
spell_check.py
--------------
Duyệt qua từng review trong file CSV, gọi LLM (OpenAI hoặc Together AI) để:
  1. Phát hiện lỗi chính tả / thiếu dấu tiếng Việt trong review
  2. Sửa lỗi chính tả, thêm dấu còn thiếu
  3. Giữ nguyên nghĩa, giọng điệu, từ lóng có chủ ý, tiếng Anh xen tiếng Việt

Chỉ xử lý những review thực sự (cột human_verified == TRUE hoặc Long_verified == TRUE),
bỏ qua các dòng là spam / không phải review.

Kết quả ghi vào cột mới: review_content_corrected
Nếu review không có lỗi hoặc là spam thì sao chép nguyên gốc.

Sử dụng:
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --provider openai
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --provider openai --test 10
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --provider openai --model gpt-4o
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --provider openai --output data/final_data/result.csv

    # Lọc theo is_review:
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --filter true     # chỉ is_review=TRUE (mặc định)
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --filter false    # chỉ is_review=FALSE
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --filter all      # tất cả các dòng

    # Dùng Together AI:
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --provider together
    python src/spell_check.py data/final_data/data_final_sorted_cleaned.csv --provider together --model meta-llama/Llama-3.3-70B-Instruct-Turbo

Cấu trúc output CSV thêm các cột:
    review_content_corrected : review đã sửa chính tả
    spell_has_error          : true/false — có phát hiện lỗi không
    spell_changes            : mô tả ngắn các thay đổi (nếu có)
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Vietnamese spell-checker specialized in informal employee reviews.
Your task is to fix ONLY spelling errors and missing Vietnamese tone marks (dấu) in the given text.

Strict rules:
1. Fix genuine misspellings and restore missing Vietnamese diacritics (e.g. "cong ty" → "công ty", "van hoa" → "văn hóa", "cong ty tuyet voi" → "công ty tuyệt vời").
2. Do NOT change intentional slang, informal language, or abbreviations (e.g. "oke", "hbt", "jd", "cv", "hr", "wc", "f0", "1k", "btw"). This includes Vietnamese teen/informal slang such as "hong", "hem", "hổng", "bik", "thik", "dzậy", "lun", "r", "ko", "k", "dc", "cx" — do NOT correct these to their formal equivalents (e.g. do NOT change "hong" → "không", "hem" → "không", "r" → "rồi").
3. Do NOT change Southern Vietnamese dialect spellings or pronunciation variants. Examples: "pà con" (= bà con), "dzậy" (= vậy), "gì dzậy" (= gì vậy), "vô" (= vào), "mần" (= làm). These are intentional regional forms, NOT spelling errors.
4. Do NOT change words that are correct Vietnamese but less common. Examples: "chỉn chu" (= cẩn thận, neat), "thoải mái", "hào phóng". Do NOT silently replace them with a different word.
5. When in doubt whether a word is a typo or intentional slang/dialect — leave it unchanged.
6. Do NOT change English words or brand names.
7. Add spaces for words that are stuck together (e.g., "môitrường" → "môi trường", "đồăn" → "đồ ăn", "vanphong" → "văn phòng").
8. Do NOT alter the meaning, sentence structure, tone, or style of the review.
9. Do NOT add or remove sentences or punctuation beyond what is necessary for the correction.
10. If the text has NO spelling errors at all, set has_error to false and return an EMPTY STRING "" as corrected_text.

Return a single JSON object with exactly these fields:
{
  "has_error": <true if any spelling error was found and fixed, false otherwise>,
  "corrected_text": "<full corrected review text if has_error is true, otherwise empty string>",
  "changes": "<concise description of corrections in Vietnamese, e.g. 'Thêm dấu cho: cong ty → công ty, nghi → nghĩ'. Empty string if no changes.>"
}

Output ONLY the JSON object. No explanation, no markdown, no extra text.
"""

USER_TEMPLATE = """\
Review text:
{review}
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "spell_check_errors.log"),
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPEN_CLAUDE_API_URL = "https://open-claude.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL = "gpt-5"
OPEN_CLAUDE_DEFAULT_MODEL = "chatgpt5.4"

def call_open_claude(prompt: str, model: str, api_key: str, retries: int = 4, timeout: int = 120) -> str:
    if not api_key:
        raise ValueError("OpenClaude API key chưa được cung cấp. Dùng biến môi trường OPEN_CLAUDE_API_KEY.")
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
            resp = requests.post(OPEN_CLAUDE_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
                logger.error("OpenClaude 429 rate-limit | attempt=%d | wait=%.1fs", attempt, wait)
                if attempt <= retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenClaude rate-limit vượt giới hạn sau {retries + 1} lần")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 * attempt, 30))
    raise RuntimeError(f"OpenClaude API thất bại sau {retries + 1} lần: {last_error}")

def call_openai(prompt: str, model: str, api_key: str, retries: int = 4, timeout: int = 120) -> str:
    if not api_key:
        raise ValueError("OpenAI API key chưa được cung cấp. Dùng biến môi trường OPENAI_API_KEY.")
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
            resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
                logger.error("OpenAI 429 rate-limit | attempt=%d | wait=%.1fs", attempt, wait)
                if attempt <= retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenAI rate-limit vượt giới hạn sau {retries + 1} lần")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 * attempt, 30))
    raise RuntimeError(f"OpenAI API thất bại sau {retries + 1} lần: {last_error}")


# ---------------------------------------------------------------------------
# Together AI  (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_DEFAULT_MODEL = "openai/gpt-oss-120b"


def call_together(prompt: str, model: str, api_key: str, retries: int = 4, timeout: int = 120) -> str:
    if not api_key:
        raise ValueError("Together AI API key chưa được cung cấp. Dùng biến môi trường TOGETHER_API_KEY.")
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
        "temperature": 0.5,
        "top_p": 1.0,
        "stream": False,
    }
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
                logger.error("Together 429 rate-limit | attempt=%d | wait=%.1fs", attempt, wait)
                if attempt <= retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Together rate-limit vượt giới hạn sau {retries + 1} lần")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 * attempt, 30))
    raise RuntimeError(f"Together AI API thất bại sau {retries + 1} lần: {last_error}")


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
def parse_result(raw: str, original_text: str) -> dict:
    default = {
        "review_content_corrected": original_text,
        "spell_has_error": False,
        "spell_changes": "",
    }
    try:
        json_str = extract_json_str(raw)
        data = json.loads(json_str)
    except Exception as exc:
        logger.error("JSON parse error: %s | raw=%r", exc, raw[:300])
        default["spell_has_error"] = "PARSE_ERROR"
        default["spell_changes"] = "PARSE_ERROR"
        return default

    has_error = data.get("has_error", False)
    if isinstance(has_error, str):
        has_error = has_error.lower() == "true"
    default["spell_has_error"] = bool(has_error)

    if not has_error:
        # Không có lỗi: dùng thẳng original, không lấy corrected_text từ LLM
        default["review_content_corrected"] = data.get("corrected_text", "").strip()
    else:
        corrected = data.get("corrected_text", "").strip()
        default["review_content_corrected"] = corrected if corrected else original_text

    changes = data.get("changes", "")
    default["spell_changes"] = str(changes).strip() if changes else ""

    return default


# ---------------------------------------------------------------------------
# Process a single row
# ---------------------------------------------------------------------------
def process_row(
    review_text: str,
    model: str,
    provider: str = "openai",
    openai_api_key: str = "",
    together_api_key: str = "",
    open_claude_api_key: str = "",
) -> dict:
    review = str(review_text).strip() if pd.notna(review_text) else ""

    if not review:
        return {
            "review_content_corrected": "",
            "spell_has_error": False,
            "spell_changes": "",
        }

    user_prompt = USER_TEMPLATE.format(review=review)
    try:
        if provider == "together":
            raw = call_together(user_prompt, model=model, api_key=together_api_key)
        elif provider == "open_claude":
            raw = call_open_claude(user_prompt, model=model, api_key=open_claude_api_key)
        else:
            raw = call_openai(user_prompt, model=model, api_key=openai_api_key)
        return parse_result(raw, original_text=review)
    except Exception as exc:
        logger.error("Lỗi API | provider=%s | review=%r | error=%s", provider, review[:80], exc)
        return {
            "review_content_corrected": "ERROR",
            "spell_has_error": "ERROR",
            "spell_changes": "ERROR",
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
    parser = argparse.ArgumentParser(description="Kiểm tra và sửa chính tả review tiếng Việt bằng LLM")
    parser.add_argument("csv_file", help="Đường dẫn tới file CSV đầu vào")
    parser.add_argument("--test", type=int, default=0, metavar="N",
                        help="Chỉ chạy N dòng đầu để kiểm tra")
    parser.add_argument("--rerun", action="store_true",
                        help="Xử lý lại các dòng đã có kết quả")
    parser.add_argument("--filter", default="true", choices=["all", "true", "false"],
                        help="Lọc theo cột is_review: 'true' (mặc định), 'false', hoặc 'all'")
    parser.add_argument("--provider", default="openai", choices=["openai", "together", "open_claude"],
                        help="LLM provider: 'openai' (mặc định) hoặc 'together' hoặc 'open_claude'")
    parser.add_argument("--model", default=None,
                        help=(
                            "Tên model. Mặc định theo provider: "
                            "openai=gpt-4o-mini | together=meta-llama/Llama-3.3-70B-Instruct-Turbo"
                        ))
    parser.add_argument("--delay", type=float, default=0.5, metavar="SEC",
                        help="Thời gian chờ (giây) giữa mỗi request để tránh rate limit (mặc định: 0.5)")
    parser.add_argument("--output", default=None, metavar="CSV_PATH",
                        help="Đường dẫn file CSV output (mặc định: <input>_spellchecked.csv)")
    args = parser.parse_args()

    import os
    load_dotenv()

    DEFAULT_MODELS = {
        "openai": OPENAI_DEFAULT_MODEL,
        "together": TOGETHER_DEFAULT_MODEL,
        "open_claude": OPEN_CLAUDE_DEFAULT_MODEL,
    }
    if args.model is None:
        args.model = DEFAULT_MODELS.get(args.provider, OPENAI_DEFAULT_MODEL)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    together_api_key = os.environ.get("TOGETHER_API_KEY", "")
    open_claude_api_key = os.environ.get("OPEN_CLAUDE_API_KEY", "")

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[LỖI] Không tìm thấy file: {csv_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else csv_path.with_name(csv_path.stem + "_spellchecked.csv")

    # Resume từ checkpoint nếu có
    if not args.rerun and output_path.exists():
        print(f"Tìm thấy file checkpoint, tiếp tục từ: {output_path}")
        df = pd.read_csv(output_path, dtype=str, encoding="utf-8-sig")
    else:
        print(f"Đọc file: {csv_path}")
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    print(f"  → {len(df)} dòng, {len(df.columns)} cột: {list(df.columns)}")

    # Tìm cột review
    col_review = find_col(df, ["review_content_cleaned", "review_content", "review"])
    if col_review is None:
        print(f"[LỖI] Không tìm thấy cột review. Các cột hiện có: {list(df.columns)}")
        sys.exit(1)
    print(f"  → Cột review: '{col_review}'")

    # Khởi tạo cột output nếu chưa có
    for col in ["review_content_corrected", "spell_has_error", "spell_changes"]:
        if col not in df.columns:
            df[col] = ""

    # Xác định dòng cần xử lý
    # 1. Lọc theo --filter (cột is_review)
    col_is_review = find_col(df, ["is_review"])
    if args.filter == "all":
        valid_mask = pd.Series([True] * len(df))
        print("  → Chế độ: xử lý TẤT CẢ các dòng")
    elif col_is_review is None:
        valid_mask = pd.Series([True] * len(df))
        print("  [CẢNH BÁO] Không tìm thấy cột is_review, sẽ xử lý tất cả các dòng")
    else:
        target = "TRUE" if args.filter == "true" else "FALSE"
        valid_mask = df[col_is_review].astype(str).str.strip().str.upper() == target
        print(f"  → Lọc: {col_is_review} == {target} | {valid_mask.sum()} / {len(df)} dòng")

    # 2. Dòng verified chưa xử lý → gọi LLM
    if args.rerun:
        needs_llm = valid_mask.copy()
    else:
        done_mask = (
            df["spell_has_error"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])
        )
        needs_llm = valid_mask & ~done_mask

    # Dòng không verified, chưa có review_content_corrected → chỉ copy, không gọi LLM
    not_done_corrected = df["review_content_corrected"].astype(str).str.strip().isin(["", "nan"])
    needs_copy = ~valid_mask & not_done_corrected

    # Gộp lại, rồi mới apply --test limit lên toàn bộ
    all_to_handle = needs_llm | needs_copy

    if args.test > 0:
        test_indices = df[all_to_handle].head(args.test).index
        all_to_handle = pd.Series(False, index=df.index)
        all_to_handle[test_indices] = True
        needs_llm = needs_llm & all_to_handle
        needs_copy = needs_copy & all_to_handle
        print(f"\n[TEST MODE] Xử lý {all_to_handle.sum()} dòng (LLM: {needs_llm.sum()}, copy: {needs_copy.sum()}).\n")
    else:
        print(f"\nSẽ xử lý {all_to_handle.sum()} dòng (LLM: {needs_llm.sum()}, copy: {needs_copy.sum()}).\n")

    all_rows_to_handle = df[all_to_handle].sort_index().index.tolist()

    if len(all_rows_to_handle) == 0:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ Đã lưu: {output_path}")
        print("Không có dòng nào cần xử lý. Dùng --rerun để chạy lại.")
        sys.exit(0)

    print(f"File output: {output_path}")

    # Kiểm tra kết nối provider
    if args.provider == "openai":
        print(f"\nKiểm tra kết nối OpenAI (model: {args.model}) ...")
        if not openai_api_key:
            print("  [LỖI] Chưa có OpenAI API key. Đặt biến môi trường OPENAI_API_KEY.")
            sys.exit(1)
        try:
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            print(f"  ✓ Kết nối OpenAI OK.")
        except Exception as exc:
            print(f"  [CẢNH BÁO] Không kiểm tra được OpenAI: {exc}")
    elif args.provider == "together":
        print(f"\nKiểm tra kết nối Together AI (model: {args.model}) ...")
        if not together_api_key:
            print("  [LỖI] Chưa có Together AI API key. Đặt biến môi trường TOGETHER_API_KEY.")
            sys.exit(1)
        try:
            resp = requests.get(
                "https://api.together.xyz/v1/models",
                headers={"Authorization": f"Bearer {together_api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            print(f"  ✓ Kết nối Together AI OK.")
        except Exception as exc:
            print(f"  [CẢNH BÁO] Không kiểm tra được Together AI: {exc}")

    # Vòng lặp xử lý
    errors = 0
    has_errors_count = 0
    copied_count = 0
    SAVE_EVERY = 10

    print()
    for i, idx in enumerate(tqdm(all_rows_to_handle, desc="Spell-check", unit="dòng")):
        if needs_llm[idx]:
            result = process_row(
                review_text=df.at[idx, col_review],
                model=args.model,
                provider=args.provider,
                openai_api_key=openai_api_key,
                together_api_key=together_api_key,
                open_claude_api_key=open_claude_api_key,
            )
            if args.delay > 0:
                time.sleep(args.delay)
            if result["spell_has_error"] == "ERROR":
                errors += 1
            elif result["spell_has_error"] is True:
                has_errors_count += 1
        else:
            # Dòng bị bỏ qua: copy nguyên gốc, không gọi LLM
            orig = df.at[idx, col_review]
            result = {
                "review_content_corrected": str(orig) if pd.notna(orig) else "",
                "spell_has_error": "False",
                "spell_changes": "",
            }
            copied_count += 1

        df.at[idx, "review_content_corrected"] = result["review_content_corrected"]
        df.at[idx, "spell_has_error"] = str(result["spell_has_error"])
        df.at[idx, "spell_changes"] = result["spell_changes"]

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Lưu kết quả cuối
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    llm_count = needs_llm.sum()
    print(f"\n✓ Đã lưu: {output_path}")
    print(f"  Tổng dòng xử lý         : {len(all_rows_to_handle)}")
    print(f"    - Gọi LLM             : {llm_count}")
    print(f"    - Copy nguyên bản     : {copied_count}")
    print(f"  Phát hiện có lỗi CT     : {has_errors_count}")
    print(f"  Không có lỗi             : {llm_count - has_errors_count - errors}")
    print(f"  Lỗi API                  : {errors}")
    if errors:
        print(f"  Chi tiết lỗi             : logs/spell_check_errors.log")

    # Thống kê mẫu: in ra vài dòng đã sửa
    corrected_df = df.loc[needs_llm & (df["spell_has_error"].astype(str).str.upper() == "TRUE")]
    if not corrected_df.empty:
        print(f"\n--- Ví dụ {min(5, len(corrected_df))} dòng đã sửa ---")
        for _, row in corrected_df.head(5).iterrows():
            print(f"\n  [GỐC]   {str(row[col_review])[:120]}")
            print(f"  [SỬA]   {str(row['review_content_corrected'])[:120]}")
            print(f"  [THAY]  {row['spell_changes']}")


if __name__ == "__main__":
    main()
