"""
llm_process.py
--------------
Duyệt qua từng review trong file CSV, gọi LLM (OpenAI hoặc OpenClaude) để:
  1. Trích xuất các khía cạnh (aspect) và cảm xúc tương ứng (sentiment)

LLM trả về JSON → parse → ghi ra CSV.

Sử dụng:
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider openai
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider openai --test 10
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider openai --model gpt-4o
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider openai --output result.csv

    # Dùng OpenClaude:
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider open_claude
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider open_claude --model chatgpt5.4

    # Dùng ChatGPT Plus (local proxy):
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider chatgpt_plus
    python src/absa_llm_process.py data/final_data/data_final_sorted_cleaned.csv --provider chatgpt_plus --model gemini-claude-opus-4-5-thinking

Cấu trúc output CSV thêm các cột:
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
    "Process & Policies",
    "HR & Recruitment",
    "Project & Technology",
    "Job Security",
    "General",
]

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

SYSTEM_PROMPT = """\
You are an expert analyzer for Vietnamese employee reviews.
Reviews are written in Vietnamese or mixed Vietnamese-English (informal, with slang and teencode).

Your job for each review is to return a single JSON object with exactly these fields:

{
  "aspects": [
    {"aspect": "<aspect label>", "sentiment": "<positive|negative|neutral>"}
  ]
}

Rules:
1.Aspects: Extract only the aspects that are explicitly or clearly implied in the review. Each aspect must be one of the following (use the exact label):
    - Salary & Benefits: Mentions of monthly pay, 13th-month salary, performance bonuses, insurance, or perks (lunch, gym, health check-ups).
    - Management & Leadership: Comments on the technical competence, leadership style, transparency, or attitude of managers, CEOs, or team leads.
    - Culture & Environment: Specific comments about interpersonal relationships, teamwork, social dynamics, or the "personality" of the company (e.g., teamwork vs. toxic drama), and diversity.
    - Work Hours & Workload: Mentions of official hours, overtime (OT) frequency, burnout, work pressure, or work-life balance (WLB).
    - Career Growth & Opportunities: Discussion about promotion tracks, career ladders, performance reviews, and long-term professional growth.
    - Process & Policies: Feedback on internal workflows, administrative bureaucracy, project management methodologies (Agile/Scrum), and company-wide rules.
    - Training & Learning: Mentions of technical workshops, mentorship programs, support for certifications, or the quality of the onboarding process.
    - Office & Workspace: Physical infrastructure, office location, interior design, and provided hardware (laptops, dual monitors, chairs).
    - HR & Recruitment: The quality of interaction with HR, the interview experience, speed of the hiring process, and the job offer stage.
    - Project & Technology: The nature of the stack (e.g., legacy vs. modern), complexity of the tasks, project scale, and the domain (Fintech, E-commerce, etc.).
    - Job Security: Stability of the position, clarity of labor contracts, or concerns regarding layoffs and downsizing.
    - General: Broad statements that describe the overall experience without pointing to any specific reason or department (e.g., "Good place to work") that cannot be mapped to any of the 11 categories above.

2. Sentiment: For each aspect, assign one of the following sentiments based on the reviewer's tone and content:
    - Positive: Clear satisfaction, praise, or highlighting a strength (e.g., "rất tốt", "tuyệt vời", "hài lòng").
    - Negative: Clear dissatisfaction, complaints, or highlighting a weakness (e.g., "tệ", "hãm", "toxic", "thất vọng").
    - Neutral: You MUST assign Neutral if the review meets any of the following criteria:
        + Mixed Sentiment: The reviewer mentions both positive and negative points for the SAME aspect (e.g., "Lương cao nhưng thưởng hơi ít" -> Aspect: Salary, Sentiment: Neutral).
        + Indifferent/Mediocre Language: Use of words that lack clear emotion or indicate an "average" state. Key Vietnamese phrases include: "bình thường", "tạm được", "cũng được", "không quá...", "không hẳn", "cũng ổn", "đủ sống", "chấp nhận được".
        + Objective Facts: Purely descriptive statements without any subjective evaluation (e.g., "Công ty dùng Java" -> Aspect: Technology, Sentiment: Neutral).
        + Example: "Lương cũng được, mọi thứ ở mức chấp nhận được đối với một lập trình viên mới ra trường." → aspect: "Salary & Benefits" (neutral), aspect: "General" (neutral),
                "Khối lượng công việc không ít không nhiều nhưng nhìn chung là vẫn đảm bảo tiến độ." → aspect: "Work Hours & Workload" (neutral),
If no aspects apply, return an empty list: "aspects": []

Output ONLY the JSON object. No explanation, no markdown, no extra text.
"""

USER_TEMPLATE = """\
Review:
{review}
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename=str(PROJECT_ROOT / "logs" / "absa_llm_process_errors.log"),
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
CHATGPT_PLUS_API_URL = "http://localhost:8317/v1/chat/completions"
OPENAI_DEFAULT_MODEL = "gpt-5"
OPEN_CLAUDE_DEFAULT_MODEL = "chatgpt5.4"
CHATGPT_PLUS_DEFAULT_MODEL = "gpt-5.2"
CHATGPT_PLUS_DEFAULT_API_KEY = "proxypal-local"


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
                logger.error(
                    "OpenClaude 429 rate-limit | attempt=%d | wait=%.1fs | body=%s",
                    attempt, wait, resp.text[:500],
                )
                if attempt <= retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenClaude rate-limit vượt giới hạn sau {retries + 1} lần | body={resp.text[:500]}")
            if not resp.ok:
                logger.error(
                    "OpenClaude HTTP %d | attempt=%d | body=%s",
                    resp.status_code, attempt, resp.text[:500],
                )
                resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 * attempt, 30))
    raise RuntimeError(f"OpenClaude API thất bại sau {retries + 1} lần: {last_error}")


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
def process_row(
    review_text: str,
    company_name: str,
    model: str,
    provider: str = "openai",
    openai_api_key: str = "",
    open_claude_api_key: str = "",
    chatgpt_plus_api_key: str = CHATGPT_PLUS_DEFAULT_API_KEY,
) -> dict:
    review = str(review_text).strip() if pd.notna(review_text) else ""
    company = str(company_name).strip() if pd.notna(company_name) else ""

    if not review:
        return {
            "aspects_raw": "[]",
            "aspect_labels": "",
            "aspect_sentiments": "",
        }

    user_prompt = USER_TEMPLATE.format(company_name=company, review=review)
    try:
        if provider == "open_claude":
            raw = call_open_claude(user_prompt, model=model, api_key=open_claude_api_key)
        elif provider == "chatgpt_plus":
            raw = call_chatgpt_plus(user_prompt, model=model, api_key=chatgpt_plus_api_key)
        else:
            raw = call_openai(user_prompt, model=model, api_key=openai_api_key)
        return parse_result(raw, fallback_text=review)
    except Exception as exc:
        logger.error("Lỗi API | provider=%s | company=%r | review=%r | error=%s", provider, company, review[:80], exc)
        return {
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
    parser.add_argument("csv_file", help="Đường dẫn tới file CSV (.csv)")
    parser.add_argument("--test", type=int, default=0, metavar="N",
                        help="Chỉ chạy N dòng đầu để kiểm tra")
    parser.add_argument("--rerun", action="store_true",
                        help="Xử lý lại các dòng đã có kết quả")
    parser.add_argument("--provider", default="openai", choices=["openai", "open_claude", "chatgpt_plus"],
                        help="LLM provider: 'openai' (mặc định), 'open_claude', hoặc 'chatgpt_plus' (local proxy)")
    parser.add_argument("--model", default=None,
                        help=(
                            "Tên model. Mặc định theo provider: "
                            f"openai={OPENAI_DEFAULT_MODEL} | open_claude={OPEN_CLAUDE_DEFAULT_MODEL} | chatgpt_plus={CHATGPT_PLUS_DEFAULT_MODEL}"
                        ))
    parser.add_argument("--delay", type=float, default=0.5, metavar="SEC",
                        help="Thời gian chờ (giây) giữa mỗi request để tránh rate limit (mặc định: 0.5)")
    parser.add_argument("--output", default=None, metavar="CSV_PATH",
                        help="Đường dẫn file CSV output")
    args = parser.parse_args()

    import os
    load_dotenv()

    DEFAULT_MODELS = {
        "openai": OPENAI_DEFAULT_MODEL,
        "open_claude": OPEN_CLAUDE_DEFAULT_MODEL,
        "chatgpt_plus": CHATGPT_PLUS_DEFAULT_MODEL,
    }
    if args.model is None:
        args.model = DEFAULT_MODELS.get(args.provider, OPENAI_DEFAULT_MODEL)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    open_claude_api_key = os.environ.get("OPEN_CLAUDE_API_KEY", "")
    chatgpt_plus_api_key = os.environ.get("CHATGPT_PLUS_API_KEY", CHATGPT_PLUS_DEFAULT_API_KEY)

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[LỖI] Không tìm thấy file: {csv_path}")
        sys.exit(1)

    # Output path — xác định sớm để kiểm tra resume
    output_path = Path(args.output) if args.output else csv_path.with_name(csv_path.stem + "_processed.csv")

    if not args.rerun and output_path.exists():
        print(f"Tìm thấy file checkpoint, tiếp tục từ: {output_path}")
        df = pd.read_csv(output_path, dtype=str, encoding="utf-8-sig")
    else:
        print(f"Đọc file: {csv_path}")
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    print(f"  → {len(df)} dòng, {len(df.columns)} cột: {list(df.columns)}")

    # Tìm cột review và company
    col_review = find_col(df, ["review_content_cleaned", "review_content_masked", "review_content", "review", "combined_review"])
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
    for col in ["aspect_labels", "aspect_sentiments", "aspects_raw"]:
        if col not in df.columns:
            df[col] = ""

    # Chỉ xử lý các dòng is_review == TRUE
    is_review_mask = df["is_review"].astype(str).str.strip().str.upper() == "TRUE"
    print(f"  → is_review == TRUE: {is_review_mask.sum()} / {len(df)} dòng")

    # Xác định dòng cần xử lý
    if args.rerun:
        mask = is_review_mask
    else:
        not_done = df["aspect_labels"].isna() | (df["aspect_labels"].astype(str).str.strip().isin(["", "ERROR", "None"]))
        mask = is_review_mask & not_done

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
    elif args.provider == "open_claude":
        print(f"\nKiểm tra kết nối OpenClaude (model: {args.model}) ...")
        if not open_claude_api_key:
            print("  [LỖI] Chưa có OpenClaude API key. Đặt biến môi trường OPEN_CLAUDE_API_KEY.")
            sys.exit(1)
        try:
            resp = requests.get(
                "https://open-claude.com/v1/models",
                headers={"Authorization": f"Bearer {open_claude_api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            print(f"  ✓ Kết nối OpenClaude OK.")
        except Exception as exc:
            print(f"  [CẢNH BÁO] Không kiểm tra được OpenClaude: {exc}")
    elif args.provider == "chatgpt_plus":
        print(f"\nDùng ChatGPT Plus local proxy tại {CHATGPT_PLUS_API_URL} (model: {args.model}) ...")

    # Vòng lặp xử lý
    rows_to_process = df[mask].index.tolist()
    errors = 0
    SAVE_EVERY = 10  # lưu checkpoint sau mỗi N dòng

    print()
    for i, idx in enumerate(tqdm(rows_to_process, desc="Xử lý", unit="dòng")):
        result = process_row(
            review_text=df.at[idx, col_review],
            company_name=df.at[idx, col_company],
            model=args.model,
            provider=args.provider,
            openai_api_key=openai_api_key,
            open_claude_api_key=open_claude_api_key,
            chatgpt_plus_api_key=chatgpt_plus_api_key,
        )

        if args.delay > 0:
            time.sleep(args.delay)
        df.at[idx, "aspect_labels"] = result["aspect_labels"]
        df.at[idx, "aspect_sentiments"] = result["aspect_sentiments"]
        df.at[idx, "aspects_raw"] = result["aspects_raw"]

        if result["aspect_labels"] == "ERROR":
            errors += 1

        # Checkpoint: lưu định kỳ để không mất dữ liệu nếu máy tắt giữa chừng
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Lưu kết quả
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu: {output_path}")
    print(f"  Tổng dòng xử lý   : {len(rows_to_process)}")
    print(f"  Lỗi API            : {errors}")
    if errors:
        print(f"  Chi tiết lỗi       : logs/llm_process_errors.log")

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
