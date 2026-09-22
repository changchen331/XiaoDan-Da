"""语料准备：原始文档 → 解析 → 清洗 → 切分（带磁盘缓存）。

为什么单独成模块：三条链路必须看到**完全一致**的 chunk 才能互相印证——
`scripts/build_index.py` 建索引、`scripts/synthesize_rag_eval.py` 合成评测集、
`evaluation/chunk_experiment.py` 做切分对照实验。
评测集里记录"这条问题出自哪个 chunk"，正是为了在切分策略改变后
仍能精确计算 Recall@10；若三处各写一份解析/切分逻辑，随时间漂移后
评测集记录的来源 chunk 会与索引里的 chunk 对不上，该指标随即失效。

缓存设计：解析是整条链路最贵的一步（单篇 9.2MB 培养方案 fast 解析需 333s），
但其结果只取决于文件内容，故按 (文件大小, 修改时间) 做指纹缓存**清洗后的正文**；
切分则每次现场执行——切分对照实验需要在同一份正文上跑多种切分策略。
"""

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from knowledge_base.preprocessing.chunker import chunk_by_type
from knowledge_base.preprocessing.cleaner import INVISIBLE_CHAR_RE, clean_text
from knowledge_base.preprocessing.parser import elements_to_text, parse_document

# 支持的文档扩展名：白名单之外的内容不进入管线。
# 必须与爬虫的 ATTACHMENT_RE 保持一致——爬虫下载了但此处不认的格式会被静默忽略，
# 形成「文件躺在 data/raw 里却永远进不了索引」的隐蔽缺口
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".html", ".htm", ".doc", ".docx", ".txt", ".md")

# 目录名 → doc_type 映射；未命中的目录一律按"手册"策略处理
DOC_TYPE_BY_DIR: dict = {"手册": "手册", "通知": "通知", "表格": "表格"}

# 通知类 txt 的元数据头部行模式：【键】值
# 适用对象 / 生效学期 由爬虫写入，是检索侧人群过滤与时效过滤的唯一依据，
# 必须在此解析出来，否则 build_filter_expr 的过滤条件永远筛不掉任何文档
NOTICE_META_RE = re.compile(r"^【(标题|发布日期|原文链接|适用对象|生效学期)】(.*)$")

# 爬虫下载的附件文件名形如「通知标题_1a2b3c4d.pdf」，尾部 8 位哈希用于避免同名冲突。
# 无元数据头部的文件以文件名作为标题，需剥掉这段哈希，否则标题会带上无意义后缀
ATTACHMENT_HASH_SUFFIX_RE = re.compile(r"_[0-9a-f]{8}$")

# 解析结果缓存：键为相对路径，值为 {size, mtime, doc_type, topic_tag, text, metadata}
CORPUS_CACHE_PATH = os.path.join("data", "processed", "parsed_corpus.json")

# 解析并行度：整库建索引时解析是纯粹的 CPU 瓶颈（单篇 9.2MB 培养方案 fast 需 333s），
# 而文件之间互不依赖，按文件切分到多个进程即可近似线性加速。
# 取 4 是内存与收益的折中——单篇大 PDF 解析峰值可占数百 MB。
PARSE_WORKERS = 4


def load_document(file_path: str, doc_type: str, topic_tag: str) -> dict:
    """加载单篇文档：通知类先解析元数据头部，其余走 Unstructured 解析。

    :param file_path: 文档路径
    :param doc_type: 文档类型（由所在目录名决定）
    :param topic_tag: 主题标签（由顶层目录名决定，用于管理后台分类）
    :return: {"text": 清洗后正文, "metadata": 文档级元数据}
    """
    file_name = os.path.basename(file_path)

    # 通知类 txt 先解析头部元数据（爬虫产出的【标题】【发布日期】等）
    header_meta: dict = {}
    if doc_type == "通知" and file_name.endswith(".txt"):
        header_meta, body_text = _parse_notice_file(file_path)
    else:
        # 固定使用 fast 策略：本次入库的都是文本型 PDF（制度文档、培养方案）。
        # 不能沿用默认的 "auto" —— 它对扫描件/图片型 PDF 会自动转 OCR 分支，
        # 而 OCR 依赖外部二进制 poppler，未安装时直接抛 PDFInfoNotInstalledError
        # 中断整个建库。fast 对图片型 PDF 只会产出较少文本，不会失败
        elements = parse_document(file_path, strategy="fast")
        body_text = elements_to_text(elements)

    # 元数据默认值：手动上传文档默认全员可见、长期有效；
    # 爬虫产出的通知带【适用对象】【生效学期】头部，优先采用头部值
    metadata: dict = {
        "source_url": header_meta.get("原文链接", "manual_upload"),
        "doc_type": doc_type,
        "topic_tag": topic_tag,
        "target_audience": header_meta.get("适用对象", "全部"),
        "valid_semester": header_meta.get("生效学期", "长期有效"),
        "publish_date": header_meta.get("发布日期", ""),
        "language": "zh",
        "title": header_meta.get("标题") or _title_from_filename(file_name),
    }
    return {"text": clean_text(body_text), "metadata": metadata}


def load_document_chunks(
    file_path: str, doc_type: str, topic_tag: str
) -> list:
    """加载单篇文档并按其类型切分为 chunk。

    :return: [{"text", "metadata"}, ...]，与 build_index 入库的内容逐字一致
    """
    document = load_document(file_path, doc_type, topic_tag)
    return chunk_by_type(document["text"], doc_type, document["metadata"])


def iter_document_records(
    data_dir: str = os.path.join("data", "raw"), workers: int = PARSE_WORKERS
) -> tuple:
    """遍历数据目录下所有受支持文档，返回解析 + 清洗后的记录（带缓存）。

    :param data_dir: 原始数据根目录，其一级子目录名决定 doc_type
    :param workers: 解析并行度（进程数）；1 表示串行
    :return: (records, failures)
        - records: [{"path", "doc_type", "topic_tag", "text", "metadata"}, ...]
          按 path 排序，保证同一批文档无论缓存命中多少，输出顺序都一致
          （评测集的抽样带固定随机种子，顺序一变抽样结果就变，评测集将不可复现）
        - failures: [(文件名, 异常描述), ...] 单个文件失败不中断整批
    """
    if not os.path.isdir(data_dir):
        return [], []

    cache = _load_cache()
    records: list = []
    pending: list = []
    cached_hits = 0

    for dir_name in sorted(os.listdir(data_dir)):
        sub_dir = os.path.join(data_dir, dir_name)
        if not os.path.isdir(sub_dir):
            continue

        doc_type = DOC_TYPE_BY_DIR.get(dir_name, "手册")
        for file_name in sorted(os.listdir(sub_dir)):
            file_path = os.path.join(sub_dir, file_name)
            if not os.path.isfile(file_path) or not file_name.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):
                continue

            # 缓存键用相对路径：整目录搬移后缓存依然有效
            cache_key = os.path.relpath(file_path).replace("\\", "/")
            fingerprint = _fingerprint(file_path)
            cached = cache.get(cache_key)
            if cached and cached.get("fingerprint") == fingerprint:
                records.append(dict(cached["record"]))
                cached_hits += 1
                continue
            pending.append((cache_key, file_path, fingerprint, doc_type, dir_name))

    failures: list = []
    for cache_key, parsed in _iter_parsed(pending, workers):
        if "error" in parsed:
            failures.append((os.path.basename(cache_key), parsed["error"]))
            print(f"[corpus] 跳过（解析失败）{os.path.basename(cache_key)}: {parsed['error'][:120]}")
            continue
        cache[cache_key] = parsed
        records.append(parsed["record"])
        _save_cache(cache)  # 逐篇落盘：整批解析以小时计，中断不应丢已得成果

    records.sort(key=lambda record: record["path"])
    print(
        f"[corpus] 文档 {len(records)} 篇（缓存命中 {cached_hits} 篇，"
        f"本次解析 {len(records) - cached_hits} 篇），失败 {len(failures)} 篇"
    )
    return records, failures


def _iter_parsed(pending: list, workers: int):
    """逐个产出解析结果：串行或并行，两种方式都保证单文件失败不中断整批。

    为什么用进程而不是线程：unstructured 的 PDF 解析是纯 Python 循环，
    线程受 GIL 约束拿不到并行度；而各文件之间彼此独立，天然适合按文件切分到进程。
    缓存的读写只发生在主进程——多个子进程同时重写同一个 JSON 会互相覆盖。
    """
    if not pending:
        return
    if workers > 1 and len(pending) > 1:
        try:
            yield from _iter_parsed_parallel(pending, workers)
            return
        except (BrokenProcessPool, OSError) as pool_error:
            # Windows 的 spawn 要求父进程的 __main__ 是一个真实文件：
            # 交互式解释器与 python -c/stdin 下建池必然失败。
            # 此时退回串行，而不是让整批解析随之作废
            print(
                f"[corpus] 进程池不可用（{type(pool_error).__name__}: {pool_error}），"
                f"改为串行解析"
            )

    for task in pending:
        yield _parse_file_task(task)


def _iter_parsed_parallel(pending: list, workers: int):
    """多进程解析：结果按提交顺序产出，每条完成即打印进度。"""
    print(f"[corpus] 并行解析 {len(pending)} 篇文档（{workers} 进程）")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for cache_key, parsed in executor.map(_parse_file_task, pending):
            status = parsed.get("error") or f"{len(parsed['record']['text'])} 字符"
            print(f"[corpus] 已解析 {os.path.basename(cache_key)}（{status[:60]}）")
            yield cache_key, parsed


def _parse_file_task(task: tuple) -> tuple:
    """解析单个文件并返回缓存记录（不触碰缓存文件本身，可安全地放进子进程）。

    :param task: (缓存键, 文件路径, 指纹, doc_type, topic_tag)
    :return: (缓存键, {"fingerprint", "record"} 或 {"error"})
    """
    cache_key, file_path, fingerprint, doc_type, topic_tag = task
    try:
        document = load_document(file_path, doc_type, topic_tag)
    except Exception as error:  # 单个文件失败不应让整批解析作废
        return cache_key, {"error": f"{type(error).__name__}: {error}"}
    return cache_key, {
        "fingerprint": fingerprint,
        "record": {
            "path": cache_key,
            "doc_type": doc_type,
            "topic_tag": topic_tag,
            "text": document["text"],
            "metadata": document["metadata"],
        },
    }


def _fingerprint(file_path: str) -> dict:
    """文件指纹：大小 + 修改时间（以秒取整，规避不同文件系统的精度差异）。"""
    stat = os.stat(file_path)
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}


# 匹配用的空白字符（含全角空格与各类换行）
_WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def normalize_for_match(text: str) -> str:
    """归一化用于跨切分策略定位：去掉全部空白字符。"""
    return _WHITESPACE_RE.sub("", text)


def locate_span(haystack: str, needle: str) -> str | None:
    """在文本中定位一段原文摘录，返回其在 haystack 中的**逐字切片**。

    存在的意义是让"评测集的证据片段能否在某个 chunk 中找到"成为一个
    可判定的问题——切分策略改变后 chunk 边界随之改变，
    只有能重新定位到证据，Recall@10 才谈得上精确计算。

    空白不敏感：LLM 摘录原文时经常把换行、缩进改写成空格，
    逐字比对会把这类正确摘录误判为"不在原文中"。

    :return: haystack 中的逐字切片；定位失败返回 None
    """
    if not needle.strip():
        return None
    if needle in haystack:
        return needle

    index_map = [i for i, char in enumerate(haystack) if not _WHITESPACE_RE.match(char)]
    flat_haystack = "".join(char for char in haystack if not _WHITESPACE_RE.match(char))
    flat_needle = normalize_for_match(needle)
    position = flat_haystack.find(flat_needle)
    if position == -1:
        return None
    return haystack[index_map[position] : index_map[position + len(flat_needle) - 1] + 1]


def _load_cache() -> dict:
    """读取解析缓存；文件缺失或损坏时返回空缓存（回退为全量重新解析）。"""
    if not os.path.exists(CORPUS_CACHE_PATH):
        return {}
    try:
        with open(CORPUS_CACHE_PATH, encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        print(f"[corpus] 缓存 {CORPUS_CACHE_PATH} 不可读，本次全量重新解析")
        return {}


def _save_cache(cache: dict) -> None:
    """写回解析缓存（先写临时文件再替换，避免中断留下半截 JSON）。"""
    os.makedirs(os.path.dirname(CORPUS_CACHE_PATH), exist_ok=True)
    temp_path = CORPUS_CACHE_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False)
    os.replace(temp_path, CORPUS_CACHE_PATH)


def _title_from_filename(file_name: str) -> str:
    """从文件名推导文档标题（无元数据头部的文件，如爬虫下载的附件）。

    依次剥掉扩展名、爬虫附加的去重哈希后缀，以及标题本身可能残留的文档后缀
    与零宽字符，最终得到干净的通知标题。
    """
    stem = INVISIBLE_CHAR_RE.sub("", os.path.splitext(file_name)[0])
    stem = ATTACHMENT_HASH_SUFFIX_RE.sub("", stem)
    # 标题自带扩展名的情形（如「…培养方案（医学）.pdf_1a2b3c4d.pdf」）再剥一层
    return (
        os.path.splitext(stem)[0]
        if os.path.splitext(stem)[1].lower() in SUPPORTED_EXTENSIONS
        else stem
    )


def _parse_notice_file(file_path: str) -> tuple:
    """解析通知类 txt 的元数据头部，返回 (元数据, 正文)。

    头部行格式：【标题】xxx / 【发布日期】xxx / 【原文链接】xxx，
    首个空行之后的内容视为正文。
    """
    with open(file_path, encoding="utf-8") as file:
        lines = file.read().split("\n")

    meta: dict = {}
    body_start = 0
    for i, line in enumerate(lines):
        matched = NOTICE_META_RE.match(line.strip())
        if matched:
            meta[matched.group(1)] = matched.group(2).strip()
        elif line.strip() == "" and meta:
            body_start = i + 1  # 元数据头部后的首个空行是正文起点
            break

    body = "\n".join(lines[body_start:])
    return meta, body