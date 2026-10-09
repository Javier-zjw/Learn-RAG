"""
parsing.legacy —— 旧版 Office 格式（.doc / .xls / .ppt / .rtf）的转换适配器。

python-docx / openpyxl / python-pptx 只认 OOXML（docx / xlsx / pptx），
而企业存量文档里 OLE2 二进制格式仍然大量存在。LibreOffice 的无头转换是
这类格式最稳定的通用方案：把它当作外部工具，先转换成 OOXML，再复用现有的
三个解析器 —— 本模块只做"格式翻译"，不重复实现任何解析逻辑。

两个工程细节（批处理场景必须处理，否则"偶尔失败"很难排查）：
  1. 每次转换使用独立的 UserInstallation 目录。LibreOffice 的用户配置有锁，
     多进程并发转换时会因为抢锁而随机失败，独立目录彻底消除这个互相干扰；
  2. 转换在独立临时目录中进行，成功后目录整体清理，不污染原文件所在目录。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element

# 源后缀 -> 转换目标格式。目标格式名恰好就是对应解析器在注册表里的名字，
# 因此不需要第二张映射表：转出来的 docx 直接交给 parser:docx 处理。
_TARGETS: dict[str, str] = {
    ".doc": "docx",
    ".rtf": "docx",
    ".xls": "xlsx",
    ".ppt": "pptx",
}


@registry.register("parser", "libreoffice")
class LibreOfficeParser(DocumentParser):
    """旧版 Office -> OOXML -> 现有解析器。需要系统安装 LibreOffice（soffice 命令）。"""

    def __init__(self, command: str = "soffice", timeout: int = 600) -> None:
        self.command = command
        self.timeout = timeout

    def parse(self, path: Path) -> list[Element]:
        suffix = path.suffix.lower()
        target_format = _TARGETS.get(suffix)
        if target_format is None:
            raise ValueError(f"libreoffice 解析器不支持 {suffix or '无后缀'}，支持：{sorted(_TARGETS)}")

        with tempfile.TemporaryDirectory(prefix="learn-rag-convert-") as out:
            out_dir = Path(out)
            # 每次转换独立的用户配置，避免并发时抢配置锁
            profile = out_dir / "profile"
            profile.mkdir()
            cmd = [
                self.command, "--headless", "--norestore",
                f"-env:UserInstallation={profile.as_uri()}",
                "--convert-to", target_format,
                "--outdir", str(out_dir),
                str(path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self.timeout)
            except FileNotFoundError:
                raise RuntimeError(
                    f"找不到 LibreOffice 命令 '{self.command}'。旧版 Office 格式需要先安装 LibreOffice，"
                    "或把文件另存为 docx / xlsx / pptx 后再导入"
                ) from None
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"LibreOffice 转换 {path.name} 超时（>{self.timeout}s），已跳过") from None
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"LibreOffice 转换 {path.name} 失败：{(exc.stderr or '').strip()[-500:]}") from None

            converted = out_dir / f"{path.stem}.{target_format}"
            if not converted.exists():
                raise RuntimeError(f"LibreOffice 声称转换成功，但没有生成 {converted.name}")
            # 走注册表而不是直接 import 具体解析器：符合"组件都从注册表装配"的约定，
            # 也让测试和配置可以用自己的实现替换转换后的解析步骤
            return registry.build("parser", {"type": target_format}).parse(converted)
