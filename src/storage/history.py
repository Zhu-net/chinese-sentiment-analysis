# -*- coding: utf-8 -*-
"""
历史记录与反馈纠错 —— SQLite 唯一存储层
=========================================
数据契约（schema 即权威，API/前端列名均与此处对齐）：

history 表：
    id                 主键
    created_at         ISO8601 时间
    task_type          emotion | binary | long_text
    input_text         原文（截断）
    predicted_label    预测标签英文键（happy/...；二分类存 正面/负面）
    predicted_label_cn  预测标签中文
    confidence         置信度
    result_json        完整预测结果 JSON
    corrected_label    用户纠正后的标签英文键；NULL=未纠正
    corrected_at       纠正时间

设计说明：
- 短连接 + WAL 模式，FastAPI 与 Streamlit 多进程访问同一 db 文件安全
- 非破坏性：只有建表（IF NOT EXISTS），不做任何默认清库
"""
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_config

_CFG = load_config()
DB_PATH = Path(_CFG["app"]["db_path"])
MAX_INPUT_LEN = int(_CFG["app"]["max_input_store_len"])
EXPORT_DIR = Path(_CFG["app"]["feedback"]["export_dir"])

# 情绪英文键 -> (label_id, 中文名)，与 train.csv 契约对齐
_LABEL2ID = {name: i for i, name in enumerate(_CFG["emotion"]["labels"])}
_LABEL2CN = dict(zip(_CFG["emotion"]["labels"], _CFG["emotion"]["labels_cn"]))
_VALID_LABELS = set(_CFG["emotion"]["labels"])


def _connect() -> sqlite3.Connection:
    """打开一个短连接（WAL 允许多进程并发读写）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL");
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db():
    """建表（幂等，不删数据）"""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at         TEXT NOT NULL,
                task_type          TEXT NOT NULL,
                input_text         TEXT NOT NULL,
                predicted_label    TEXT,
                predicted_label_cn TEXT,
                confidence         REAL,
                result_json        TEXT NOT NULL,
                corrected_label    TEXT,
                corrected_at       TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at DESC)")
        conn.commit()


def add_record(task_type: str, input_text: str, result: dict) -> int:
    """存入一条分析记录，返回新记录 id。

    result 中约定字段（emotion/binary 两种形态都兼容）：
        emotion / emotion_cn / confidence   或  label / confidence
    长文本（long_text）取 summary 中的主导情绪。
    """
    init_db()
    text = str(input_text)[:MAX_INPUT_LEN]
    now = datetime.now().isoformat(timespec="seconds")

    if task_type == "long_text":
        summary = result.get("summary", {})
        pred_label = summary.get("dominant_emotion_en")
        pred_label_cn = summary.get("dominant_emotion")
        confidence = summary.get("dominant_ratio")
    elif task_type == "binary":
        pred_label = result.get("label")
        pred_label_cn = result.get("label")
        confidence = result.get("confidence")
    else:  # emotion
        pred_label = result.get("emotion")
        pred_label_cn = result.get("emotion_cn")
        confidence = result.get("confidence")

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO history
               (created_at, task_type, input_text, predicted_label,
                predicted_label_cn, confidence, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (now, task_type, text, pred_label, pred_label_cn,
             confidence, json.dumps(result, ensure_ascii=False)),
        )
        conn.commit()
        return cur.lastrowid


def list_records(limit: int = 50, task_type: str = None, corrected_only: bool = False) -> list:
    """查询历史（最新在前），返回字典列表（字段与前端 DataFrame 契约一致）"""
    init_db()
    sql = "SELECT * FROM history WHERE 1=1"
    params = []
    if task_type:
        sql += " AND task_type = ?"
        params.append(task_type)
    if corrected_only:
        sql += " AND corrected_label IS NOT NULL"
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def update_feedback(record_id: int, correct_label: str) -> bool:
    """写入/更新用户纠正标签（英文键），返回是否更新成功"""
    init_db()
    if correct_label not in _VALID_LABELS:
        raise ValueError(f"非法标签: {correct_label}，允许: {sorted(_VALID_LABELS)}")
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE history SET corrected_label = ?, corrected_at = ? WHERE id = ?",
            (correct_label, now, int(record_id)),
        )
        conn.commit()
        return cur.rowcount > 0


def export_feedback_dataframe() -> pd.DataFrame:
    """导出所有已纠正样本。

    仅导出 emotion/long_text 任务、且纠正标签与原预测不同的记录
    （纠正成相同值没有训练价值）。输出列与 processed_emotion/train.csv 完全对齐：
    text,label,label_name,label_cn,source
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT input_text, predicted_label, corrected_label FROM history
               WHERE corrected_label IS NOT NULL
                 AND task_type IN ('emotion', 'long_text')"""
        ).fetchall()

    data = []
    for r in rows:
        name = r["corrected_label"]
        # 只导出"原预测 ≠ 纠正标签"的样本；跳过非法标签
        if name not in _LABEL2ID or name == r["predicted_label"]:
            continue
        data.append({
            "text": r["input_text"],
            "label": _LABEL2ID[name],
            "label_name": name,
            "label_cn": _LABEL2CN[name],
            "source": "user_feedback",
        })
    return pd.DataFrame(data, columns=["text", "label", "label_name", "label_cn", "source"])


def export_feedback_csv() -> tuple[Path, int]:
    """导出纠错样本 CSV（utf-8-sig 便于 Excel 打开），返回 (路径, 条数)"""
    df = export_feedback_dataframe()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"feedback_{stamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path, len(df)


def stats() -> dict:
    """历史统计（给前端指标卡用）"""
    init_db()
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM history").fetchone()["c"]
        corrected = conn.execute(
            "SELECT COUNT(*) c FROM history WHERE corrected_label IS NOT NULL"
        ).fetchone()["c"]
    return {"total": total, "corrected": corrected}
