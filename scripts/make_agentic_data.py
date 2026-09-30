"""
生成 Agentic 演示数据：带元数据的文件、SQLite 库、带路由标注的问答集。

关键在于**问答集里标了 expected_tools**：
只有标了"这道题本来该用哪个工具",才能评测"自主决策"这一层对不对。
这是 agentic RAG 评测与传统 RAG 评测最大的区别。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"

# ---- 带元数据的文件：year / popularity / doc_kind 用于元数据过滤与加权排序 ----
FILES = [
    {"id": "policy_2024", "title": "2024年报销政策", "file_type": "text", "year": 2024, "popularity": 95,
     "doc_kind": "政策",
     "text": "2024年报销政策规定：差旅住宿标准一线城市每晚上限600元，其余城市400元。"
             "餐饮补贴每人每天120元。所有报销需在费用发生后30天内提交，超期不予受理。"},
    {"id": "policy_2019", "title": "2019年报销政策（已废止）", "file_type": "text", "year": 2019, "popularity": 10,
     "doc_kind": "政策",
     "text": "2019年报销政策规定：差旅住宿标准一线城市每晚上限400元，其余城市250元。"
             "餐饮补贴每人每天80元。报销需在费用发生后60天内提交。本政策自2024年起废止。"},
    {"id": "guide_rag", "title": "RAG 使用手册", "file_type": "text", "year": 2025, "popularity": 60,
     "doc_kind": "手册",
     "text": "本手册介绍检索增强生成的使用方法。混合检索建议同时开启向量与BM25两个通道，"
             "融合使用RRF算法，常数k取60。精排推荐使用CrossEncoder模型。"},
    {"id": "report_q3", "title": "第三季度经营报告", "file_type": "text", "year": 2025, "popularity": 75,
     "doc_kind": "报告",
     "text": "第三季度公司实现营业收入1250万元，同比增长18%。其中华东区贡献480万元，"
             "华南区贡献390万元，华北区贡献380万元。三季度新签客户42家。"},
    {"id": "guide_expense", "title": "报销系统操作教程", "file_type": "text", "year": 2025, "popularity": 88,
     "doc_kind": "教程",
     "text": "登录报销系统后，在左侧菜单选择新建报销单，上传发票图片，系统会自动识别金额。"
             "填写费用类型和事由后提交，由直属主管审批，财务复核后三个工作日内到账。"},
]

# ---- 结构化数据：给 sql_query 工具用 ----
ORDERS = [
    (1, "华东", "A产品", 120, 45.5),
    (2, "华南", "B产品", 80, 120.0),
    (3, "华东", "B产品", 200, 120.0),
    (4, "华北", "A产品", 60, 45.5),
    (5, "华南", "C产品", 150, 88.0),
    (6, "华东", "C产品", 90, 88.0),
]

# ---- 带路由标注的问答集 ----
QA = [
    {"id": "a1", "question": "现在的差旅住宿标准一线城市是多少？", "answers": ["600元"],
     "gold_doc_ids": ["policy_2024"], "expected_tools": ["meta_filter"],
     "gold_weights": {"policy_2024": 3.0, "policy_2019": 0.5},
     "note": "含时效性偏好：必须选新政策，纯语义检索会把两版政策混排"},
    {"id": "a2", "question": "最新的餐饮补贴标准是每天多少？", "answers": ["120元"],
     "gold_doc_ids": ["policy_2024"], "expected_tools": ["meta_filter"],
     "gold_weights": {"policy_2024": 3.0, "policy_2019": 0.5}},
    {"id": "a3", "question": "RRF 融合的常数 k 取多少？", "answers": ["60"],
     "gold_doc_ids": ["guide_rag"], "expected_tools": ["kb_search"]},
    {"id": "a4", "question": "报销单提交后由谁审批？", "answers": ["直属主管"],
     "gold_doc_ids": ["guide_expense"], "expected_tools": ["kb_search"]},
    {"id": "a5", "question": "订单表里一共有多少条记录？", "answers": ["6"],
     "gold_doc_ids": [], "expected_tools": ["sql_query"],
     "note": "结构化统计：检索文本永远答不对"},
    {"id": "a6", "question": "统计各区域的订单总金额并排名", "answers": ["华东", "华南", "华北"],
     "gold_doc_ids": [], "expected_tools": ["sql_query"]},
    {"id": "a7", "question": "你好", "answers": ["你好"],
     "gold_doc_ids": [], "expected_tools": ["final_answer"],
     "note": "寒暄：正确行为是**不检索**，传统管线会无脑检索一次"},
    {"id": "a8", "question": "第三季度华东区营收占总营收的百分比是多少？", "answers": ["38.4%", "38.4"],
     "gold_doc_ids": ["report_q3"], "expected_tools": ["kb_search", "python_exec"],
     "note": "先取数再计算：单靠 LLM 心算不可靠"},
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    with (ROOT / "agentic_files.jsonl").open("w", encoding="utf-8") as fh:
        for row in FILES:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    db_path = ROOT / "agentic_demo.db"
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, product TEXT, qty INTEGER, price REAL)"
        )
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", ORDERS)

    with (ROOT / "agentic_qa.jsonl").open("w", encoding="utf-8") as fh:
        for row in QA:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已生成：{ROOT/'agentic_files.jsonl'}、{db_path}、{ROOT/'agentic_qa.jsonl'}")


if __name__ == "__main__":
    main()
