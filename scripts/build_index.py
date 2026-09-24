"""知识库索引构建脚本：原始文档 → 解析 → 清洗 → 切分 → 向量化 → 入库。

用法：
    python -m scripts.build_index [数据目录] [--rebuild]

注：必须用 `python -m` 从仓库根目录运行（直接执行 `scripts/build_index.py` 时
仓库根不在 `sys.path` 上，`knowledge_base` 等顶层包会 import 失败）。

流水线实现与幂等语义见 ``knowledge_base.indexing.build_index``（本脚本只负责
命令行参数与数据契约说明，逻辑放在模块层以便单测覆盖与第二入口复用）。

数据目录约定（默认 data/raw/，目录名决定 doc_type 切分策略）：
    data/raw/
    ├── 手册/                  # 学生手册、培养方案等长文档（PDF/DOCX/TXT/MD）
    │   └── 学生手册.pdf
    ├── 通知/                  # 官网通知（爬虫产出或手动收集，TXT/HTML）
    │   └── <url_hash>.txt     #   爬虫产出自带【标题】【发布日期】头部
    ├── 表格/                  # 校历、时间表等表格类文档（PDF/HTML）
    │   └── 校历.pdf
    └── faq.json               # FAQ 标准问答库（结构见下）

faq.json 格式（数组，每项一个问答条目）：
    [
      {
        "question": "校园卡丢了怎么办？",
        "answer": "请携带有效证件到一卡通中心挂失补办……",
        "similar_questions": ["校园卡挂失流程", "饭卡丢了去哪里补"],
        "tags": ["校园卡"]
      }
    ]

通知类 txt 的元数据头部格式（爬虫自动产出，手动收集时照此编写）：
    【标题】关于2026年春季学期选课安排的通知
    【发布日期】2026-01-15
    【原文链接】https://jwc.fudan.edu.cn/xx/xxxx.htm
    【适用对象】全部
    【生效学期】2026-2027秋季
其中【适用对象】【生效学期】是检索侧人群过滤与时效过滤的唯一数据来源，
缺失时按"全部 / 长期有效"处理（即不做任何过滤）。

每次知识库更新后重跑本脚本，随后运行 evaluation/ragas_eval.py
确认检索质量无退化。
"""

import argparse

from knowledge_base.indexing.build_index import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建知识库索引（默认按来源幂等重建，--rebuild 整库重建）"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/raw",
        help="原始数据根目录（默认 data/raw）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="整库重建：先删除两个 Collection（schema / 向量维度变更时必须使用）",
    )
    arguments = parser.parse_args()
    build_index(arguments.data_dir, rebuild=arguments.rebuild)
