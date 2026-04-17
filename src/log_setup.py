"""
log_setup.py
------------
Cấu hình logging tập trung cho toàn project.
Tất cả scripts import và gọi setup_logging() trước khi dùng logger.

Ví dụ:
    from log_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
"""

import logging
import logging.config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
_CONFIG_PATH = Path(__file__).parent / "logging.ini"


def setup_logging() -> None:
    """Load logging config từ logging.ini, tạo thư mục logs/ nếu chưa có."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        logging.config.fileConfig(
            f,
            defaults={"logs_dir": str(LOGS_DIR).replace("\\", "/")},
            disable_existing_loggers=False,
        )
