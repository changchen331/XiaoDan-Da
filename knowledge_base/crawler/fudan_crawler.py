"""数据采集：复旦官网增量爬虫（每周定时执行，抓取新增通知 / 政策页面）。

数据源规划（对应架构文档模块一的六类来源）：
- 教务类：教务处官网（选课、考试、学籍政策）
- 研究生类：研究生院官网（培养、学位、奖助）
- 学工类：学生工作部、书院官网（奖惩、助困、活动）
- 生活服务类：一卡通中心、保卫处、总务处（校园生活）
- 学术类：图书馆官网（借阅、讲座）
- 线下 PDF（学生手册、新生指南）：不走爬虫，直接放入 data/raw/ 对应目录

增量机制：以 URL 的 MD5 为去重标识，已抓取的 URL 记录在
data/processed/crawled_urls.json 中，重复爬取时自动跳过。

产出格式：每篇通知保存为一个 txt 文件，头部三行为结构化元数据：
    【标题】关于2026年春季学期选课安排的通知
    【发布日期】2026-01-15
    【原文链接】https://jwc.fudan.edu.cn/xx/xxxx.htm
之后为正文内容。该格式可被 build_index 管线直接解析。
"""
import hashlib
import json
import os
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# 数据源配置：列表页 URL → 该源全部文档的元数据
# target_audience 决定检索时的人群过滤（全部 / 本科生 / 研究生 / …）
DATA_SOURCES: dict = {
    "https://jwc.fudan.edu.cn/": {
        "topic_tag": "教务", "target_audience": "全部", "language": "zh",
    },
    "https://gs.fudan.edu.cn/": {
        "topic_tag": "研究生院", "target_audience": "研究生", "language": "zh",
    },
    "https://www.fudan.edu.cn/": {
        "topic_tag": "综合", "target_audience": "全部", "language": "zh",
    },
}

# 浏览器 UA：避免被官网的基础反爬拦截
HEADERS: dict = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/120.0.0.0 Safari/537.36"}

# 详情页链接特征：官网通知页通常是 .htm/.html/.shtml 结尾的静态页
DETAIL_PAGE_RE = re.compile(r"\.(htm|html|shtml)$")

# 增量记录文件路径（与爬虫产出同在 data/ 目录树内）
CRAWLED_RECORD_PATH = os.path.join("data", "processed", "crawled_urls.json")

# 爬虫产出目录：以通知类型入库，doc_type 由目录名决定
OUTPUT_DIR = os.path.join("data", "raw", "通知")


class FudanCrawler:
    """复旦官网增量爬虫：列表页发现链接 → 增量去重 → 详情页抓取与结构化保存。"""

    def __init__(self, rate_limit: float = 1.0) -> None:
        """初始化爬虫。

        :param rate_limit: 相邻请求的最小间隔秒数（1 秒起步，
            对官网服务器友好，整个数据源全量抓取约需数分钟）
        """
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last_request_time: float = 0.0

    def crawl_all(self) -> list:
        """遍历所有数据源执行增量抓取。

        单个数据源失败（网络波动 / 页面改版）不影响其余数据源继续抓取。

        :return: 本次新抓取的文档摘要列表 [{"title", "path", "source"}, ...]
        """
        crawled_urls = self._load_crawled_record()
        new_documents: list = []

        for source_url, source_meta in DATA_SOURCES.items():
            try:
                new_documents.extend(
                    self._crawl_source(source_url, source_meta, crawled_urls)
                )
            except requests.RequestException as request_error:
                # 数据源级别容错：记录错误并继续下一个源
                print(f"[crawler] 数据源 {source_url} 抓取失败: {request_error}")

        self._save_crawled_record(crawled_urls)
        print(f"[crawler] 本次共新增 {len(new_documents)} 篇文档")
        return new_documents

    def _crawl_source(self, source_url: str, source_meta: dict,
                      crawled_urls: dict) -> list:
        """抓取单个数据源：列表页提取链接 → 逐篇抓取详情。

        :return: 该源新抓取的文档摘要列表
        """
        topic_tag = source_meta["topic_tag"]
        print(f"[crawler] 开始抓取数据源: {source_url} ({topic_tag})")

        # 第一步：请求列表页并解析出全部同域链接
        soup = self._fetch_page(source_url)
        detail_links = self._extract_detail_links(soup, source_url)
        print(f"[crawler] 列表页发现 {len(detail_links)} 个链接")

        # 第二步：逐篇抓取新链接（已抓取的直接跳过）
        documents: list = []
        for link in detail_links:
            url_hash = self.url_hash(link)
            if url_hash in crawled_urls:
                continue  # 增量抓取：旧文档不重复处理

            try:
                document = self._fetch_detail(link, source_meta)
                if document is None:
                    crawled_urls[url_hash] = link   # 空页面也记录，避免反复抓取
                    continue

                file_path = self._save_document(document, link, url_hash)
                crawled_urls[url_hash] = link
                documents.append({
                    "title": document["title"],
                    "path": file_path,
                    "source": link,
                })
                print(f"[crawler] 已抓取: {document['title']}")
            except requests.RequestException as detail_error:
                print(f"[crawler] 详情页失败 {link}: {detail_error}")

        return documents

    def _fetch_page(self, url: str) -> BeautifulSoup:
        """带限速的页面请求，返回解析后的 BeautifulSoup 对象。"""
        # 限速：确保两次请求间隔不小于 rate_limit 秒
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding   # 中文官网编码自适应
        self._last_request_time = time.time()
        return BeautifulSoup(response.text, "html.parser")

    def _extract_detail_links(self, list_page: BeautifulSoup, source_url: str) -> list:
        """从列表页提取详情页链接。

        规则：同域名 + 以 .htm/.html/.shtml 结尾（官网通知的标准静态页形态），
        去除锚点与查询参数后按出现顺序去重。
        """
        from urllib.parse import urljoin, urlparse

        source_domain = urlparse(source_url).netloc
        links: list = []
        seen: set = set()
        for anchor in list_page.find_all("a", href=True):
            absolute_url = urljoin(source_url, anchor["href"])
            parsed = urlparse(absolute_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            is_same_domain = parsed.netloc == source_domain
            is_detail_page = bool(DETAIL_PAGE_RE.search(parsed.path))
            if is_same_domain and is_detail_page and clean_url not in seen:
                seen.add(clean_url)
                links.append(clean_url)
        return links

    def _fetch_detail(self, url: str, source_meta: dict) -> dict | None:
        """抓取详情页并提取结构化内容。

        :return: {"title", "content", "publish_date", "source_meta"}，
            页面无有效正文时返回 None
        """
        soup = self._fetch_page(url)

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        # 优先取正文容器（官网通知通常在 article / 正文 class 中），退化取全页文本
        body = soup.find("article") or soup.find(class_=re.compile("content|article|main")) or soup.body
        if body is None:
            return None

        content = body.get_text(separator="\n", strip=True)
        if len(content) < 50:
            return None   # 过短内容多为导航页或空壳页，无入库价值

        publish_date = self._extract_publish_date(soup.get_text(), source_meta)

        return {
            "title": title,
            "content": content,
            "publish_date": publish_date,
            "source_meta": source_meta,
        }

    @staticmethod
    def _extract_publish_date(page_text: str, source_meta: dict) -> str:
        """从页面文本中提取发布日期（官网通知的标准格式：2026-01-15）。

        未匹配到时回退为抓取日期，保证元数据字段始终有值。
        """
        date_match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", page_text)
        if date_match:
            year, month, day = date_match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _save_document(document: dict, url: str, url_hash: str) -> str:
        """将结构化文档保存为带元数据头部的 txt 文件。

        文件名使用 url_hash 保证唯一且对文件系统安全（不含非法字符）。
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"{url_hash}.txt")

        header = (
            f"【标题】{document['title']}\n"
            f"【发布日期】{document['publish_date']}\n"
            f"【原文链接】{url}\n"
        )
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(header + "\n" + document["content"])
        return file_path

    @staticmethod
    def _load_crawled_record() -> dict:
        """加载已抓取 URL 记录（增量判断依据）。"""
        if os.path.exists(CRAWLED_RECORD_PATH):
            with open(CRAWLED_RECORD_PATH, encoding="utf-8") as file:
                return json.load(file)
        return {}

    @staticmethod
    def _save_crawled_record(record: dict) -> None:
        """持久化已抓取 URL 记录。"""
        os.makedirs(os.path.dirname(CRAWLED_RECORD_PATH), exist_ok=True)
        with open(CRAWLED_RECORD_PATH, "w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

    @staticmethod
    def url_hash(url: str) -> str:
        """URL 去重标识：MD5 摘要，同时用作产出文件名。"""
        return hashlib.md5(url.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # 直接运行本模块即执行一轮增量抓取：
    #   python -m knowledge_base.crawler.fudan_crawler
    FudanCrawler().crawl_all()
