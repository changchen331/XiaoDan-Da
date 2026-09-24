"""同步边界自检：扫描"会入库的文件"里是否提到了本地专用路径。

**要防的事**：本项目把本地文档目录（迭代计划、评审登记、证据记录）与原始设计
文档排除在仓库之外，但它们仍可能被写进 README、代码注释或提交信息里。
一旦写进去，克隆仓库的人就会看到指向不存在路径的引用——观感是"内部资料没清干净"。
本项目已实际踩过两次（README 引用、忽略清单的注释），故做成可执行的自检。

**为什么读忽略清单而不是写死路径名**：忽略清单是"哪些路径不入库"的唯一权威
来源。新增本地专用目录时，只要它进了忽略清单，本脚本自动覆盖；写死路径名会漏，
而且两份清单迟早会不一致。

**为什么要"带分隔符"才算命中**：裸词歧义太大——``docstring``、
``skipped_docs`` 这类标识符、URL 里的 ``docs.astral.sh``、以及容器路径
``/app/.venv/...`` 都会命中。故目录要求写成 ``名字/``，且**前一字符**不能是
字母数字、下划线、点或斜杠；文件要求写全名。

**放行清单为什么存在**：忽略清单里还有两类路径，提到它们**不算越界**——
① 运行期布局（``models/`` ``data/`` ``volumes/`` 等挂载 / 产物目录）；
② 构建与工具产物（``build/`` ``__pycache__/`` ``venv/`` 等）。
它们只被忽略，不是"不该让外人知道的东西"；而本脚本要守的是后者
（本地文档目录与设计文档）。放行是显式决定，逐项列在 ALLOWED_NAMES 里，
不是漏判。

用法（退出码非 0 表示发现越界引用）：
    uv run python -m scripts.check_synced_boundary          # 默认取脚本所在的仓库
    uv run python -m scripts.check_synced_boundary <仓库根>  # 指定其他仓库
"""

import re
import subprocess
import sys
from pathlib import Path

#: 忽略项中**允许**被同步文件引用的名字（运行期布局 + 构建 / 工具产物）
ALLOWED_NAMES = frozenset(
    {
        # 运行期布局：挂载目录与产物目录
        "data",
        "models",
        "volumes",
        "hf_cache",
        "logs",
        "outputs",
        # 构建与工具产物
        "build",
        "dist",
        "venv",
        "env",
        "__pycache__",
        "htmlcov",
    }
)

#: 只对"文档类"忽略文件做判定：生成物（csv / log / 权重）被说明提及是正常的
DOC_SUFFIXES = (".md", ".txt", ".pdf", ".docx", ".doc")

#: 豁免文件：忽略清单自身必须写出这些路径名，否则规则不生效
EXEMPT_FILES = frozenset({".gitignore"})


def _local_only_patterns(ignore_path: Path) -> list[tuple[str, str]]:
    """从忽略清单提取"本地专用"路径名及其正则，返回 [(显示名, 模式), ...]。"""
    patterns: list[tuple[str, str]] = []
    for raw_line in ignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # 注释与反选（!）不参与；空行跳过
        if not line or line.startswith(("#", "!")):
            continue
        name = line.split("/")[0].strip()
        # 通配与工具目录（.venv / .vscode / *.py[cod] 等）不参与判定
        if not name or name.startswith(("*", ".", "!")):
            continue
        if name in ALLOWED_NAMES:
            continue

        if "/" in line:  # 目录：要求写成 "名字/"
            suffix = "/"
        elif name.lower().endswith(DOC_SUFFIXES):  # 文档类文件：要求写全名
            suffix = ""
        else:
            continue
        patterns.append(
            (f"{name}{suffix}", rf"(?<![\w./]){re.escape(name)}{re.escape(suffix)}")
        )
    return patterns


def _tracked_files(root: Path) -> list[str]:
    """取当前被 git 跟踪的文件清单。

    用 ``git ls-files`` 而不是遍历目录：遍历会把忽略项一起扫进来，
    而本脚本要判定的恰恰是"会入库的文件"。
    """
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    """扫描跟踪文件，打印越界引用；返回退出码。"""
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    else:
        root = Path(__file__).resolve().parent.parent
    patterns = _local_only_patterns(root / ".gitignore")
    if not patterns:
        print("[boundary] 忽略清单里没有需要保护的本地路径")
        return 0

    violations: list[str] = []
    for relative in _tracked_files(root):
        if Path(relative).name in EXEMPT_FILES:
            continue
        try:
            content = (root / relative).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # 二进制或读取失败：不参与文本判定
        for line_number, line in enumerate(content.splitlines(), start=1):
            for name, pattern in patterns:
                if re.search(pattern, line):
                    violations.append(
                        f"{relative}:{line_number}: 提到本地专用路径 {name!r}"
                    )

    if not violations:
        print(
            f"[boundary] 通过：{len(patterns)} 个本地专用路径名均未出现在同步文件中"
        )
        return 0

    print("[boundary] 发现越界引用（下列文件会入库，不应提到不入库的路径）：")
    for item in violations:
        print(f"  - {item}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
