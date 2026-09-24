"""FAQ 题库构建：从校内**公开** FAQ 页面抽取「问 / 答」对，产出 `data/raw/faq.json`。

为什么走"抽取 + 合成"两步，而不是让模型直接写题库：

- **答案只来自原文抽取**——FAQ 快路径命中后会**原样直返**答案、不经过 LLM 改写
  （见 `knowledge_base/faq.py` 的设计），所以答案一旦由模型生成，就等于把
  "可能编造的制度性内容"直接推给用户。故本脚本**不允许模型写答案**。
- 模型只做两件低风险的事：补 `similar_questions`（相似问法）与 `tags`（主题标签）。
  相似问法决定快路径的召回面，是题库可用性的关键；它写错了顶多是不命中，
  不会把错误信息答给用户。

数据来源均为公开页面（无需登录、不含个人隐私数据）；产出的 `data/raw/faq.json`
落在 `.gitignore` 覆盖的 `data/raw/` 下，与其余语料一样不入库。

用法：
    python scripts/build_faq.py            # 抽取 + 合成 + 写 data/raw/faq.json
    python scripts/build_faq.py --no-enrich  # 只抽取，不调模型（离线可跑）
"""

import argparse
import json
import os
import re

import requests
from bs4 import BeautifulSoup

from agent.llm_clients import chat_qwen_json, parse_json_response

# 公开 FAQ 来源：(URL, 主题标签)。全部为校内公开页面，无需登录。
FAQ_SOURCES: list = [
    {
        "url": "https://library.fudan.edu.cn/_s928/09/38/c42789a723256/page.psp",
        "tag": "学位论文",
        "name": "图书馆·学位论文提交常见问题",
    },
    {
        "url": "https://library.fudan.edu.cn/e8/30/c42799a518192/page.htm",
        "tag": "电子资源",
        "name": "图书馆·中国知网使用说明（含常见问题）",
    },
    {
        "url": "https://ecampus.fudan.edu.cn/cjwt/list.htm",
        "tag": "电子邮箱",
        "name": "信息办·云邮箱常见问题",
    },
]

OUTPUT_PATH = os.path.join("data", "raw", "faq.json")

#: 单条问答的长度上下限：太短的多半是抽取噪声（分页链接、导航项），太长的不适合做快路径答案
MIN_QUESTION_LEN = 6
MIN_ANSWER_LEN = 10
MAX_ANSWER_LEN = 800

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def fetch_text(url: str) -> str:
    """抓取页面并转成纯文本（保留段落换行，丢弃标签）。"""
    response = requests.get(url, headers=_HEADERS, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    # 这些站点的导航/页脚多为 div，不靠语义标签；按 class/id 关键字再清一遍，
    # 否则页脚链接会被拼进最后一条答案（答案会原样直返用户，噪声代价高）
    noise_pattern = re.compile(
        r"nav|menu|footer|breadcrumb|sidebar|copyright|links?", re.IGNORECASE
    )
    for tag in soup.find_all(attrs={"class": noise_pattern}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": noise_pattern}):
        tag.decompose()
    text = soup.get_text("\n")
    # 压缩空白但保留段落边界：抽取正则依赖换行判断问答边界
    lines = [re.sub(r"[ \t\u3000]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


#: 编号标记（`1、` / `2.` …）：用它把页面切成条目块。
#: 两种页面结构都能覆盖——编号与问句同行（`1、问：…？`）或编号独占一行（`1.` 换行后才是问句）
_NUMBER_MARK_RE = re.compile(r"(?m)^\s*\d+\s*[.、]\s*")

#: 剥离问 / 答标记：页面里写作「问：…答：…」，标记本身不属于内容。
#: 不锚定行首——问句带括号注释时（`…归档？（学位论文归档手续）答：…`），
#: 以首个问号切分会把 `答：` 留在答案开头，故按出现位置统一清掉
_QA_MARKER_RE = re.compile(r"(?:问|答)\s*[：:]")

#: 问句末尾的短括号注释（如"（学位论文归档手续）"）：以首个问号切分后会落进答案，
#: 但它属于问句的补充说明，不是答案内容
_TRAILING_NOTE_RE = re.compile(r"^\s*（[^）]{0,20}）\s*")

#: 剥离答案尾部残留的下一条编号（切块按编号边界，末块会带上下一条的 `2、`）
_TRAILING_NUMBER_RE = re.compile(r"\s*\d+\s*[.、]\s*$")


def _clean(text: str) -> str:
    """规整问答文本：去掉问/答标记、尾部编号与问句括号注释，压掉换行与多余空格。"""
    text = _QA_MARKER_RE.sub("", text)
    text = _TRAILING_NOTE_RE.sub("", text)
    collapsed = re.sub(r"\s*\n\s*", " ", text)
    collapsed = re.sub(r"\s{2,}", " ", collapsed).strip(" 　-—")
    return _TRAILING_NUMBER_RE.sub("", collapsed).strip()


def _is_valid(question: str, answer: str) -> bool:
    """过滤抽取噪声：长度上下限 + 问题须是疑问句 + 答案不能是纯链接。"""
    if not (MIN_QUESTION_LEN <= len(question) <= 120):
        return False
    if not (MIN_ANSWER_LEN <= len(answer) <= MAX_ANSWER_LEN):
        return False
    if not question.rstrip().endswith(("？", "?")):
        return False
    # 答案若几乎只剩 URL，属于导航项而非问答
    return len(re.sub(r"https?://\S+", "", answer).strip()) >= MIN_ANSWER_LEN


def extract_pairs(text: str) -> list:
    """从页面文本抽取「问 / 答」对。

    做法：先按编号标记切块，再在每块内以**第一个问号**为界——
    问号之前是问句、之后是答案。这样"编号与问句同行"和"编号独占一行"
    两种页面结构用同一套逻辑处理，不需要为每个站点写一份解析器。
    """
    pairs: list = []
    for block in _NUMBER_MARK_RE.split(text):
        mark = block.find("？")
        if mark == -1:
            mark = block.find("?")
        if mark == -1:
            continue
        question = _clean(block[: mark + 1])
        answer = _clean(block[mark + 1 :])
        if _is_valid(question, answer):
            pairs.append({"question": question, "answer": answer})
    return pairs


#: 让模型只补"相似问法 + 标签"，绝不改写答案（答案原样直返用户）
_ENRICH_PROMPT = """下面是从校园官网 FAQ 里抽出的问答。请为每条补充：
1. similar_questions：3 条**学生真实会怎么问**的相似问法（口语化，不要照抄原问句）
2. tags：1-2 个主题标签（如"图书馆""选课""邮箱"）

只输出 JSON，不要任何其他文字：
{{"items": [{{"index": 0, "similar_questions": ["...", "...", "..."], "tags": ["..."]}}]}}

问答列表：
{items}"""


def enrich(pairs: list, batch_size: int = 5) -> None:
    """用轻量模型补 similar_questions 与 tags（原地修改）。失败时保留空列表。"""
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        listing = "\n".join(
            f"{index}. 问：{item['question']}\n   答：{item['answer'][:200]}"
            for index, item in enumerate(batch)
        )
        try:
            result = parse_json_response(
                chat_qwen_json(_ENRICH_PROMPT.format(items=listing))
            )
            for entry in result.get("items", []):
                position = int(entry.get("index", -1))
                if 0 <= position < len(batch):
                    batch[position]["similar_questions"] = [
                        str(question) for question in entry.get("similar_questions", [])
                    ][:3]
                    batch[position]["tags"] = [
                        str(tag) for tag in entry.get("tags", [])
                    ][:2]
        except Exception as enrich_error:  # 单批失败不中断整轮，保留空标签
            print(f"[build_faq] 第 {start // batch_size + 1} 批相似问法生成失败: {enrich_error}")
        print(
            f"[build_faq] 相似问法进度 {min(start + batch_size, len(pairs))}/{len(pairs)}"
        )


def build(no_enrich: bool = False) -> list:
    """主流程：抓取 → 抽取 → 去重 → 合成 → 落盘。"""
    all_pairs: list = []
    seen: set = set()

    for source in FAQ_SOURCES:
        try:
            text = fetch_text(source["url"])
        except Exception as fetch_error:
            print(f"[build_faq] 跳过 {source['name']}: {fetch_error}")
            continue
        pairs = extract_pairs(text)
        kept = 0
        for pair in pairs:
            key = re.sub(r"\W+", "", pair["question"])
            if key in seen:
                continue
            seen.add(key)
            all_pairs.append(
                {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "similar_questions": [],
                    "tags": [source["tag"]],
                }
            )
            kept += 1
        print(f"[build_faq] {source['name']}: 抽取 {len(pairs)} 条，去重后保留 {kept} 条")

    if not no_enrich and all_pairs:
        enrich(all_pairs)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(all_pairs, file, ensure_ascii=False, indent=2)
    print(f"[build_faq] 已写入 {len(all_pairs)} 条 → {OUTPUT_PATH}")
    return all_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从校内公开 FAQ 页面构建 faq.json")
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="只抽取、不调模型补相似问法（离线可跑）",
    )
    arguments = parser.parse_args()
    build(no_enrich=arguments.no_enrich)
