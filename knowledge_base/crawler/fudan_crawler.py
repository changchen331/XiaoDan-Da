"""数据采集：复旦官网增量爬虫（每周定时执行，抓取新增通知 / 政策页面）。

数据源规划（对应架构文档模块一的六类来源）：
- 研究生类：研究生院官网（招生、培养、学位、奖助）
- 生活服务类：总务处官网（后勤服务、住房、食堂、物业）
- 线下 PDF（学生手册、新生指南）：不走爬虫，直接放入 data/raw/ 对应目录
- 教务类（教务处官网）：列表页公开，但详情页需复旦统一身份认证登录（实测渲染后
  为登录页，列表页也不含附件直链），正文与附件均不可得。仅列表标题入库对检索
  无实际价值，故不纳入数据源
- 学工类 / 学术类：图书馆与团委站点列表模板差异较大，待单独适配

正文获取的三条路径（实站探测结论，互为补充）：
1. 网页正文：栏目详情页是静态 HTML 且正文完整时，直接解析保存为 txt
2. 附件正文：官网大量通知的正文以 PDF 附件形式发布（页面上只有标题与发布信息），
   此时下载附件存入 data/raw/手册/，由 build_index 走长文档递归切分
3. 渲染兜底：对 JS 动态渲染的 SPA 站点，由 render_detail 开关切换到
   Playwright 无头浏览器渲染后再取 DOM（需先执行 playwright install chromium）
   注意：渲染只能解决"前端不直出 HTML"的问题，解决不了"内容需登录"的问题

增量机制：以 URL 的 MD5 为去重标识，已抓取的 URL 记录在
data/processed/crawled_urls.json 中，重复爬取时自动跳过。附件 URL 同样入表。

分页机制：栏目列表页为 list.htm，后续页为 list2.htm / list3.htm ……，
从「尾页」链接解析总页数，再由 max_pages 截断上限（避免首次全量抓取上千篇）。

条目提取（模板无关）：复旦各院系官网同属一套 CMS，但列表容器 class 各不相同
（实测：教务处 table.wp_article_list_table、研究生院 span.Article_Title、
总务处 div.news_title），因此不硬编码容器选择器，改为
「按最近带 class/id 的祖先容器分组 → 取链接数最多的一组」：
同一内容列表的条目必然重复出现相同的 DOM 结构，而导航栏、微信矩阵等噪声区块的
链接分散在不同容器且数量少，天然被排除。

产出格式：
- 网页正文：txt 文件，头部为结构化元数据（可被 build_index 直接解析）
    【标题】关于2026年春季学期选课安排的通知
    【发布日期】2026-01-15
    【原文链接】https://jwc.fudan.edu.cn/xx/xxxx.htm
    【适用对象】全部
    【生效学期】2026-2027秋季
- 附件正文：PDF / DOCX 等原始文件，文件名即通知标题
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from config.settings import get_current_semester
from knowledge_base.preprocessing.cleaner import INVISIBLE_CHAR_RE, clean_text

# ==================== 数据源配置 ====================
# 每个源对应一个栏目列表页，URL 均经实站验证可用（HTTP 200 且能解析出条目列表）。
# target_audience：检索时的人群过滤字段（全部 / 本科生 / 研究生 / 留学生 / 教职工）
# valid_semester："长期有效"（规章制度、常用文档类）或 "发布学期"
#                 （通知、动态类，按发布月份推算学期，使过期通知被时效过滤挡掉）
# max_pages：该源最多抓取的分页数（列表页每页约 10-14 条）
# render_detail：详情页是否需 Playwright 渲染（Vue SPA 站点开启；列表页均为静态模板，
#                始终按普通 GET 抓取）。当前数据源均无需渲染，该开关作为能力保留，
#                供后续接入 SPA 站点时使用（渲染链路已在实站验证可用）。
DATA_SOURCES: tuple = (
    # ---------- 研究生类（静态详情页，正文多为 PDF 附件）----------
    {
        "name": "研究生院-管理规章",
        "list_url": "https://gs.fudan.edu.cn/2673/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "研究生院-培养方案",
        "list_url": "https://gs.fudan.edu.cn/2679/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "研究生院-学籍管理",
        "list_url": "https://gs.fudan.edu.cn/2731/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "研究生院-毕业管理",
        "list_url": "https://gs.fudan.edu.cn/2743/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "发布学期",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "研究生院-文件规定",
        "list_url": "https://gs.fudan.edu.cn/2800/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "研究生院-申请流程",
        "list_url": "https://gs.fudan.edu.cn/2803/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "研究生院-奖学金政策",
        "list_url": "https://gs.fudan.edu.cn/2863/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "研究生院-相关通知",
        "list_url": "https://gs.fudan.edu.cn/11107/list.htm",
        "topic_tag": "研究生院",
        "target_audience": "研究生",
        "valid_semester": "发布学期",
        "max_pages": 1,
        "render_detail": False,
    },
    # ---------- 生活服务类 ----------
    {
        "name": "总务处-通知公告",
        "list_url": "https://zongwuchu.fudan.edu.cn/16531/list.htm",
        "topic_tag": "生活服务",
        "target_audience": "全部",
        "valid_semester": "发布学期",
        "max_pages": 2,
        "render_detail": False,
    },
    {
        "name": "总务处-规章制度",
        "list_url": "https://zongwuchu.fudan.edu.cn/39287/list.htm",
        "topic_tag": "生活服务",
        "target_audience": "全部",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "总务处-住房服务",
        "list_url": "https://zongwuchu.fudan.edu.cn/39310/list.htm",
        "topic_tag": "生活服务",
        "target_audience": "全部",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "总务处-常用服务",
        "list_url": "https://zongwuchu.fudan.edu.cn/39306/list.htm",
        "topic_tag": "生活服务",
        "target_audience": "全部",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "总务处-食堂食品安全",
        "list_url": "https://zongwuchu.fudan.edu.cn/xsstspaqglxx/list.htm",
        "topic_tag": "生活服务",
        "target_audience": "全部",
        "valid_semester": "发布学期",
        "max_pages": 1,
        "render_detail": False,
    },
    {
        "name": "总务处-节能管理",
        "list_url": "https://zongwuchu.fudan.edu.cn/jnglgzqk/list.htm",
        "topic_tag": "生活服务",
        "target_audience": "全部",
        "valid_semester": "长期有效",
        "max_pages": 1,
        "render_detail": False,
    },
)

# 详情页 URL 特征：CMS 详情页统一为 /xx/xx/cNNNNNaNNNNN/page.htm
DETAIL_PAGE_RE = re.compile(r"page\.htm$")
# 列表页分页特征：list.htm → list2.htm → list3.htm ……
PAGE_NUM_RE = re.compile(r"list(\d+)\.htm")
# 附件链接特征：与 build_index 的 SUPPORTED_EXTENSIONS 严格对齐 —— 只下载能入库的格式。
# 若此处放宽而索引侧不认，附件会躺在 data/raw 里却永远进不了索引（静默缺口）。
# 需要支持 xlsx/pptx 时，两处与 unstructured 的 extras 必须同时补齐
ATTACHMENT_RE = re.compile(r"\.(pdf|doc|docx)$", re.IGNORECASE)
# 文件名非法字符（Windows 与 POSIX 合集）
ILLEGAL_FILENAME_RE = re.compile(r'[\\/:*?"<>|\r\n\t]')
# 列表条目文本的栏目前缀（实测研究生院为「培养 | 复旦大学……」，需剥离）
TITLE_PREFIX_RE = re.compile(r"^[^|\s]{1,12}\s*[|｜]\s*")
# 块级标签：这些标签处才换行，行内标签（span / a / font）原样拼接
BLOCK_TAGS: frozenset = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "td",
        "th",
        "table",
        "thead",
        "tbody",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "section",
        "article",
        "blockquote",
        "pre",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)

# 正文容器候选选择器：各院系模板 class 不一，逐个尝试后取文本最长者
CONTENT_SELECTORS: tuple = (
    "div.wp_articlecontent",  # 教务处 / 通用 CMS 正文容器
    "div.Article_Content",  # 研究生院
    "div.news_content",  # 总务处
    "div.article",  # 总务处文章块
    "div.content",
    "article",
)

# 列表条目最小数量：低于该值视为导航 / 推荐位区块而非内容列表
MIN_LIST_ITEMS = 3
# 网页正文最小长度：低于该值视为该页没有网页正文（官网大量通知只在附件里），
# 转由附件路径处理，避免把「标题 + 发布者 + 浏览次数」这类元信息当作正文入库
MIN_ARTICLE_LENGTH = 200
# 附件体积区间：低于下限多为占位文件或错误页；高于上限多为扫描件 / 图册，
# 解析产出文本极少却拖慢索引构建，且这类内容不属于问答知识
MIN_ATTACHMENT_BYTES = 2048
MAX_ATTACHMENT_BYTES = 30 * 1024 * 1024
# 向上寻找「带 class/id 的祖先容器」的最大层数
MAX_ANCESTOR_DEPTH = 8
# Playwright 渲染超时：SPA 需等前端拉完数据，给足 30 秒
RENDER_TIMEOUT_MS = 30000

# 浏览器 UA：避免被官网的基础反爬拦截
HEADERS: dict = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
}

# 增量记录文件路径（与爬虫产出同在 data/ 目录树内）
CRAWLED_RECORD_PATH = os.path.join("data", "processed", "crawled_urls.json")

# 产出目录：网页正文按通知类型入库（doc_type 由目录名决定）；
# 附件统一按长文档处理——制度类 PDF 篇幅长，适用递归切分策略
OUTPUT_DIR = os.path.join("data", "raw", "通知")
ATTACHMENT_OUTPUT_DIR = os.path.join("data", "raw", "手册")

# ===== Playwright 渲染后端（进程级单例，多个页面复用同一浏览器实例）=====
_playwright_runtime = None
_playwright_browser = None


def _get_browser():
    """惰性启动 Playwright Chromium。

    :raises RuntimeError: 未执行 ``playwright install chromium`` 时抛出
    """
    global _playwright_runtime, _playwright_browser
    if _playwright_browser is None:
        from playwright.sync_api import sync_playwright

        _playwright_runtime = sync_playwright().start()
        _playwright_browser = _playwright_runtime.chromium.launch(headless=True)
    return _playwright_browser


def _release_browser() -> None:
    """关闭浏览器与 Playwright 运行时（爬取结束后调用，避免进程挂起）。"""
    global _playwright_runtime, _playwright_browser
    if _playwright_browser is not None:
        _playwright_browser.close()
        _playwright_browser = None
    if _playwright_runtime is not None:
        _playwright_runtime.stop()
        _playwright_runtime = None


class FudanCrawler:
    """复旦官网增量爬虫：列表页发现条目 → 增量去重 → 详情页抓取与结构化保存。"""

    def __init__(self, rate_limit: float = 1.0) -> None:
        """初始化爬虫。

        :param rate_limit: 相邻请求的最小间隔秒数（1 秒起步，对官网服务器友好）
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

        try:
            for source in DATA_SOURCES:
                try:
                    new_documents.extend(self._crawl_source(source, crawled_urls))
                except requests.RequestException as request_error:
                    # 数据源级别容错：记录错误并继续下一个源
                    print(
                        f"[crawler] 数据源 {source['name']} 抓取失败: {request_error}"
                    )
        finally:
            # 增量记录先落盘：即使中途异常，已抓取的部分也不会被重复抓取
            self._save_crawled_record(crawled_urls)
            _release_browser()

        print(f"[crawler] 本次共新增 {len(new_documents)} 篇文档")
        return new_documents

    def _crawl_source(self, source: dict, crawled_urls: dict) -> list:
        """抓取单个数据源：逐页解析条目列表 → 逐篇抓取详情。

        :return: 该源新抓取的文档摘要列表
        """
        source_name = source["name"]
        list_url = source["list_url"]
        max_pages = int(source.get("max_pages", 1))
        render_detail = bool(source.get("render_detail", False))
        print(f"[crawler] 开始抓取数据源: {source_name} ({list_url})")

        documents: list = []
        total_pages = 1
        for page_number in range(1, max_pages + 1):
            # 总页数由首页的「尾页」链接解析得出，超出即停止翻页
            if page_number > total_pages:
                break

            page_url = (
                list_url
                if page_number == 1
                else list_url.replace("list.htm", f"list{page_number}.htm")
            )
            try:
                soup = self._fetch_page(page_url)
            except requests.RequestException as page_error:
                print(f"[crawler] 列表页失败 {page_url}: {page_error}")
                break

            if page_number == 1:
                total_pages = self._total_pages(soup, list_url)

            items = self._extract_items(soup, page_url)
            if not items:
                # 列表条目识别失败通常意味着模板变更，提前终止该源避免空转
                print(
                    f"[crawler] {source_name} 第 {page_number} 页未识别出条目列表，跳过该源"
                )
                break
            print(
                f"[crawler] {source_name} 第 {page_number}/{min(total_pages, max_pages)} 页: "
                f"{len(items)} 条"
            )

            for title, link in items:
                url_hash = self.url_hash(link)
                if url_hash in crawled_urls:
                    continue  # 增量抓取：旧文档不重复处理

                crawled_urls[url_hash] = link  # 空页面同样记录，避免反复抓取
                saved = self._save_page(link, title, source, render_detail)
                documents.extend(saved)

        return documents

    def _save_page(
        self, url: str, title: str, source: dict, render_detail: bool
    ) -> list:
        """抓取并落盘一个详情页：优先网页正文，其次附件下载。

        官网大量通知的正文只存在于附件（PDF）中，页面上仅有标题与发布信息，
        因此网页正文不足长度阈值时自动转附件路径，两条路径都拿不到则跳过。

        :return: 落盘的文档摘要列表（可能为空）
        """
        try:
            soup = self._fetch_page(url, render=render_detail)
        except Exception as page_error:
            # 渲染失败（浏览器未安装 / 超时）与网络异常都不应中断整个数据源
            print(
                f"[crawler] 详情页失败 {url}: {type(page_error).__name__} {page_error}"
            )
            return []

        content = self._extract_content(soup)
        if len(content) >= MIN_ARTICLE_LENGTH:
            document = {
                "title": title or self._page_title(soup, url),
                "content": content,
                "publish_date": self._extract_publish_date(soup.get_text(), url),
            }
            file_path = self._save_document(document, url, self.url_hash(url), source)
            print(f"[crawler] 已抓取网页正文: {document['title'][:40]}")
            return [{"title": document["title"], "path": file_path, "source": url}]

        attachments = self._extract_attachments(soup, url)
        if not attachments:
            print(f"[crawler] 无有效网页正文且无附件，跳过: {url}")
            return []

        saved_documents: list = []
        for attachment_url in attachments:
            file_path = self._download_attachment(attachment_url, title)
            if file_path is None:
                continue
            saved_documents.append(
                {
                    "title": title,
                    "path": file_path,
                    "source": attachment_url,
                }
            )
            print(f"[crawler] 已下载附件: {os.path.basename(file_path)}")
        return saved_documents

    @staticmethod
    def _total_pages(soup: BeautifulSoup, list_url: str) -> int:
        """从「尾页」链接解析栏目总页数（解析不到时按 1 页处理）。"""
        for anchor in soup.find_all("a", href=True):
            if "尾页" not in anchor.get_text(strip=True):
                continue
            matched = PAGE_NUM_RE.search(urljoin(list_url, anchor["href"]))
            if matched:
                return int(matched.group(1))
        return 1

    def _extract_items(self, soup: BeautifulSoup, list_url: str) -> list:
        """从列表页提取条目 (标题, 链接)，与模板具体 class 无关。

        实现要点：同一内容列表的条目重复出现相同的 DOM 结构，因此按
        「最近带 class/id 的祖先容器」分组后取链接数最多的一组即可命中真实列表；
        导航栏、微信矩阵等噪声区块的链接分散在不同容器且数量少，被自然排除。

        :return: [(标题, 绝对 URL), ...]；未识别出有效列表时返回空列表
        """
        groups: dict = {}
        for anchor in soup.find_all("a", href=True):
            absolute_url = urljoin(list_url, anchor["href"])
            if not DETAIL_PAGE_RE.search(urlparse(absolute_url).path):
                continue
            container = self._nearest_marked_ancestor(anchor)
            key = (
                container.name,
                tuple(container.get("class") or ()),
                container.get("id"),
            )
            groups.setdefault(key, []).append((anchor, absolute_url))

        if not groups:
            return []

        best_group = groups[max(groups, key=lambda group_key: len(groups[group_key]))]
        if len(best_group) < MIN_LIST_ITEMS:
            return []

        items: list = []
        seen_urls: set = set()
        for anchor, absolute_url in best_group:
            # 条目文本常带栏目前缀（「培养 | 复旦大学……」），剥离后才是通知标题
            title = TITLE_PREFIX_RE.sub("", anchor.get_text(strip=True)).strip()
            if title and absolute_url not in seen_urls:
                seen_urls.add(absolute_url)
                items.append((title, absolute_url))
        return items

    @staticmethod
    def _nearest_marked_ancestor(anchor: Tag) -> Tag:
        """向上寻找最近的带 class 或 id 的祖先容器（用于分组识别列表）。

        找不到时以链接自身所在标签为分组依据，保证每个链接都有归属。
        """
        node = anchor.parent
        depth = 0
        while node is not None and depth < MAX_ANCESTOR_DEPTH:
            if node.get("class") or node.get("id"):
                return node
            node = node.parent
            depth += 1
        return anchor

    def _fetch_page(self, url: str, render: bool = False) -> BeautifulSoup:
        """请求页面并返回解析后的 BeautifulSoup 对象。

        :param render: True 时用 Playwright 渲染后再取 DOM（Vue SPA 站点），
            False 时直接 GET 静态 HTML
        """
        self._wait_for_rate_limit()

        if render:
            return BeautifulSoup(self._render_page(url), "html.parser")

        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(self._decode_html(response), "html.parser")

    @staticmethod
    def _decode_html(response: requests.Response) -> str:
        """解码 HTML 字节流，字符集判定顺序：页面声明 → HTTP 头 → 内容探测。

        不能只依赖 requests 的 apparent_encoding：内容探测对中文页面会误判
        （实测总务处某页被判为 MacCyrillic，落盘文本成乱码，检索必然失效）。
        官网 CMS 页面均带 <meta charset="utf-8">，以页面自身声明最为可靠。
        """
        candidates: list = []
        meta_match = re.search(
            rb'charset=["\']?\s*([\w-]+)', response.content[:4096], re.IGNORECASE
        )
        if meta_match:
            candidates.append(meta_match.group(1).decode("ascii", errors="ignore"))
        header_match = re.search(
            r"charset=([\w-]+)", response.headers.get("Content-Type", ""), re.IGNORECASE
        )
        if header_match:
            candidates.append(header_match.group(1))
        candidates.append(response.apparent_encoding or "utf-8")

        for charset in candidates:
            try:
                # 严格模式解码：字符集猜错必然抛异常，从而自动尝试下一个候选
                return response.content.decode(charset, errors="strict")
            except (UnicodeDecodeError, LookupError):
                continue
        return response.content.decode("utf-8", errors="replace")

    @staticmethod
    def _render_page(url: str) -> str:
        """用 Playwright 渲染 JS 动态页面，返回渲染后的完整 HTML。

        等 networkidle 确保前端 XHR 拉完正文，否则拿到的是应用外壳。
        """
        browser = _get_browser()
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        try:
            page.goto(url, wait_until="networkidle", timeout=RENDER_TIMEOUT_MS)
            return page.content()
        finally:
            page.close()

    def _wait_for_rate_limit(self) -> None:
        """限速：确保两次请求间隔不小于 rate_limit 秒（对官网服务器友好）。"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    @classmethod
    def _extract_content(cls, soup: BeautifulSoup) -> str:
        """提取网页正文：在所有已知正文容器候选中取文本最长者。

        刻意不采用「全页最长 div」兜底：官网模板的导航栏往往比正文还长
        （实测总务处导航 234 字符 > 正文块 58 字符），兜底会选错容器。
        因此取不到足够长的正文时返回短文本，由调用方转入附件路径。

        提取结果经 clean_text 清洗：正文容器里混着「发布者 / 发布时间 /
        浏览次数」等网文元信息，不清洗会让这些噪声计入正文长度，
        使空壳页面被误判为有效正文而入库。
        """
        longest = ""
        for selector in CONTENT_SELECTORS:
            element = soup.select_one(selector)
            if element is None:
                continue
            text = cls._block_text(element)
            if len(text) > len(longest):
                longest = text
        return clean_text(longest)

    @classmethod
    def _block_text(cls, node: Tag) -> str:
        """按块级标签插入换行地提取纯文本。

        不能用 ``get_text(separator="\\n")``：它会在每个行内标签（span / a / font）
        处断行，把「2026年5月」切成「202 / 6 / 年 / 5月」，破坏 chunk 语义、
        显著拖累检索质量。这里只对块级标签换行，行内标签原样拼接。
        """
        chunks: list = []

        def walk(current: Tag) -> None:
            for child in current.children:
                if isinstance(child, NavigableString):
                    chunks.append(str(child))
                elif isinstance(child, Tag):
                    if child.name in BLOCK_TAGS:
                        chunks.append("\n")
                        walk(child)
                        chunks.append("\n")
                    else:
                        walk(child)

        walk(node)
        # 折叠多余空白：行内产生的连续空格压成一个，多个换行压成一个
        text = "".join(chunks)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        return re.sub(r"\n{2,}", "\n", text)

    @staticmethod
    def _extract_attachments(soup: BeautifulSoup, page_url: str) -> list:
        """提取页面中的附件链接（官网通知的真实正文常以 PDF 附件发布）。"""
        attachments: list = []
        seen_urls: set = set()
        for anchor in soup.find_all("a", href=True):
            absolute_url = urljoin(page_url, anchor["href"])
            if (
                ATTACHMENT_RE.search(urlparse(absolute_url).path)
                and absolute_url not in seen_urls
            ):
                seen_urls.add(absolute_url)
                attachments.append(absolute_url)
        return attachments

    def _download_attachment(self, url: str, title: str) -> str | None:
        """下载附件到 data/raw/手册/（长文档走递归切分策略）。

        文件名为通知标题（build_index 以文件名作为文档标题），
        后缀附加 URL 哈希片段避免同名冲突。

        :return: 落盘路径；下载失败或文件过小时返回 None
        """
        try:
            self._wait_for_rate_limit()
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            if len(response.content) < MIN_ATTACHMENT_BYTES:
                print(f"[crawler] 附件过小，跳过: {url}")
                return None
            if len(response.content) > MAX_ATTACHMENT_BYTES:
                print(
                    f"[crawler] 附件超过 {MAX_ATTACHMENT_BYTES // 1024 // 1024}MB，跳过: {url}"
                )
                return None

            extension = os.path.splitext(urlparse(url).path)[1].lower()
            file_name = (
                f"{self._safe_filename(title, 'attachment')}"
                f"_{self.url_hash(url)[:8]}{extension}"
            )
            os.makedirs(ATTACHMENT_OUTPUT_DIR, exist_ok=True)
            file_path = os.path.join(ATTACHMENT_OUTPUT_DIR, file_name)
            with open(file_path, "wb") as file:
                file.write(response.content)
            return file_path
        except (requests.RequestException, OSError) as download_error:
            print(f"[crawler] 附件下载失败 {url}: {download_error}")
            return None

    @staticmethod
    def _safe_filename(title: str, fallback: str) -> str:
        """把通知标题转为文件系统安全的文件名。

        除去掉非法字符外，还要剥掉标题末尾自带的文档后缀：研究生院列表的条目文本
        本身就含「.pdf」（如「…培养方案（人文社会科学）.pdf」），不剥离会产出
        「…社会科学）.pdf_d72402a6.pdf」这种双后缀文件名。
        同时清理零宽空格等不可见字符（官网条目文本中确实存在）。
        """
        cleaned = INVISIBLE_CHAR_RE.sub("", title)
        cleaned = ILLEGAL_FILENAME_RE.sub("", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = ATTACHMENT_RE.sub("", cleaned).strip()
        return cleaned[:60] or fallback

    @staticmethod
    def _page_title(soup: BeautifulSoup, url: str) -> str:
        """取 <title> 文本作为标题，缺失时回退为 URL。"""
        if soup.title is not None and soup.title.string:
            return soup.title.string.strip()
        return url

    @staticmethod
    def _extract_publish_date(page_text: str, fallback_url: str) -> str:
        """从页面文本中提取发布日期（官网通知的标准格式：2026-01-15）。

        未匹配到时回退为抓取日期，保证元数据字段始终有值。
        """
        date_match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", page_text)
        if date_match:
            year, month, day = date_match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        print(f"[crawler] 未解析到发布日期，回退抓取日期: {fallback_url}")
        return datetime.now().strftime("%Y-%m-%d")

    @classmethod
    def _save_document(
        cls, document: dict, url: str, url_hash: str, source: dict
    ) -> str:
        """将结构化文档保存为带元数据头部的 txt 文件。

        文件名使用 url_hash 保证唯一且对文件系统安全（不含非法字符）。
        头部字段与 build_index 的元数据解析规则一一对应，target_audience 与
        valid_semester 在此落盘，否则检索侧的人群 / 时效过滤将失效。
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"{url_hash}.txt")

        header = (
            f"【标题】{document['title']}\n"
            f"【发布日期】{document['publish_date']}\n"
            f"【原文链接】{url}\n"
            f"【适用对象】{source['target_audience']}\n"
            f"【生效学期】{cls._resolve_semester(source, document['publish_date'])}\n"
        )
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(header + "\n" + document["content"])
        return file_path

    @staticmethod
    def _resolve_semester(source: dict, publish_date: str) -> str:
        """确定文档的生效学期。

        - 制度 / 文档类：标「长期有效」，不参与时效过滤
        - 通知 / 动态类：按发布月份推算所属学期，使过期通知被时效过滤挡掉
        """
        policy = str(source.get("valid_semester", "长期有效"))
        if policy != "发布学期":
            return policy
        try:
            published = datetime.strptime(publish_date, "%Y-%m-%d")
        except ValueError:
            return "长期有效"
        return get_current_semester(published.date())

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
