"""
生成文档解析的验证样例：覆盖 MinerU 4.x 能识别的主要格式，内容短但结构复杂。

生成结果已经提交在 samples/parsing/，一般不需要重新生成；修改样例时再运行：
    python scripts/make_parsing_samples.py            # 输出到 samples/parsing/

每个文件要验证的结构和关键内容见 samples/README.md。
依赖：pymupdf、python-docx、python-pptx、openpyxl、Pillow、matplotlib；
旧版 Office（.doc/.ppt/.xls）、OpenDocument（.odt/.odp/.ods）和 .rtf 由 LibreOffice 转换生成，
没装 LibreOffice 时跳过这些格式并给出提示。中文字体使用系统里的文泉驿正黑。
"""

from __future__ import annotations

import base64
import csv
import io
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "parsing"
CJK_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
COMPANY = "星河科技"

# 各文件共用的业务数据，便于跨格式对照解析结果
REGIONS = ["华东", "华南", "华北", "西南"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
REVENUE = {  # 单位：万元
    "华东": [320, 360, 410, 480],
    "华南": [280, 300, 350, 390],
    "华北": [260, 270, 330, 380],
    "西南": [150, 170, 200, 230],
}


# ---------------------------------------------------------------------------
# 图片素材：图表、公式、流程图（matplotlib / Pillow 生成，返回 PNG 字节）
# ---------------------------------------------------------------------------
def _matplotlib():
    import logging

    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 文泉驿只有一种字重，屏蔽回退提示
    from matplotlib import font_manager, pyplot as plt

    if Path(CJK_FONT).exists():
        font_manager.fontManager.addfont(CJK_FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    _matplotlib().close(fig)
    return buf.getvalue()


def bar_chart() -> bytes:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(5, 3))
    width = 0.2
    for i, region in enumerate(REGIONS):
        ax.bar([q + i * width for q in range(4)], REVENUE[region], width, label=region)
    ax.set_xticks([q + 1.5 * width for q in range(4)], QUARTERS)
    ax.set_ylabel("营收（万元）")
    ax.set_title("2025 年各区域季度营收")
    ax.legend(fontsize=8)
    return _png(fig)


def line_chart() -> bytes:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(5, 3))
    for region in REGIONS:
        ax.plot(QUARTERS, REVENUE[region], marker="o", label=region)
    ax.set_title("季度营收趋势")
    ax.set_ylabel("万元")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _png(fig)


def formula(latex: str, size: int = 16) -> bytes:
    plt = _matplotlib()
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=size)
    return _png(fig)


def flow_diagram() -> bytes:
    """报销流程图：四个方框和箭头。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(CJK_FONT, 26)
    steps = ["提交申请", "主管审批", "财务复核", "打款到账"]
    for i, step in enumerate(steps):
        x = 20 + i * 220
        draw.rounded_rectangle([x, 60, x + 170, 140], radius=12, outline="#1f4e79", width=3, fill="#e8f0fa")
        draw.text((x + 85, 100), step, font=font, fill="black", anchor="mm")
        if i < len(steps) - 1:
            draw.line([x + 175, 100, x + 215, 100], fill="#1f4e79", width=4)
            draw.polygon([(x + 215, 100), (x + 203, 92), (x + 203, 108)], fill="#1f4e79")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _pdf_text(page, rect, text: str, size: float = 10.5, font: str = "wqy", align: int = 0) -> None:
    """写一段文字。中文用文泉驿（内置的 china-s 会把数字和字母排成全角间距，也缺上标字符），
    英文用 PDF 标准字体 tiro / tibo / cour。保存时会做字体子集化，文件不会因嵌入字体而变大。"""
    import pymupdf

    fontfile = CJK_FONT if font == "wqy" else None
    box = pymupdf.Rect(rect)
    if font == "wqy":
        text = _wrap_cjk(text, box.width - 2, size)
    left = page.insert_textbox(box, text, fontsize=size, fontname=font, fontfile=fontfile, align=align)
    if -size * 1.5 <= left < 0:      # 差不到一行：把框向下放大一点重写（放不下时 PyMuPDF 什么都不写）
        box.y1 += -left + 2
        left = page.insert_textbox(box, text, fontsize=size, fontname=font, fontfile=fontfile, align=align)
    if left < 0:
        raise ValueError(f"文本放不下（溢出 {-left:.0f}pt），请调整版面：{text[:30]}")


def _wrap_cjk(text: str, width: float, size: float) -> str:
    """
    按字符宽度自己折行。PyMuPDF 只在空格处换行，中文长句会在“2025 年”这样的空格处断出很短的行，
    和真实的中文排版（任意汉字之间都能换行、每行写满）不符，会误导解析结果的判断。
    英文单词和数字（如 4,860、18.6%）作为整体，不从中间拆开。
    """
    import re

    import pymupdf

    font = pymupdf.Font(fontfile=CJK_FONT)
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for token in re.findall(r"[A-Za-z0-9.,%/:@_+=\-]+|\s|.", paragraph):
            if line.strip() and font.text_length(line + token, fontsize=size) > width:
                lines.append(line.rstrip())
                line = token.lstrip()
            else:
                line += token
        lines.append(line.rstrip())
    return "\n".join(lines)


def _pdf_header_footer(page, number: int, total: int) -> None:
    w, h = page.rect.width, page.rect.height
    _pdf_text(page, (50, 22, w - 50, 40), f"{COMPANY} · 2025 年度经营报告 · 内部资料", size=8, align=2)
    page.draw_line((50, 42), (w - 50, 42), color=(0.6, 0.6, 0.6), width=0.5)
    _pdf_text(page, (50, h - 40, w - 50, h - 22), f"第 {number} 页 / 共 {total} 页", size=8, align=1)


def _pdf_table(page, x: float, y: float, widths: list[float], row_h: float, cells: list[tuple]) -> float:
    """cells: (行, 列, 跨行, 跨列, 文字)。按单元格画框，合并单元格就是一个大框。返回表格底部 y。"""
    import pymupdf

    rows = max(r + rs for r, _, rs, _, _ in cells)
    for r, c, rs, cs, text in cells:
        x0 = x + sum(widths[:c])
        rect = pymupdf.Rect(x0, y + r * row_h, x0 + sum(widths[c:c + cs]), y + (r + rs) * row_h)
        page.draw_rect(rect, color=(0, 0, 0), width=0.6)
        inner = pymupdf.Rect(rect.x0 + 2, rect.y0 + (rect.height - 12) / 2 - 1, rect.x1 - 2, rect.y1)
        page.insert_textbox(inner, text, fontsize=9, fontname="wqy", fontfile=CJK_FONT, align=1)
    return y + rows * row_h


def pdf_annual_report(path: Path) -> None:
    """双栏版面、多级标题、合并单元格表格、公式图、柱状图、页眉页脚、脚注、跨页段落、列表、代码。"""
    import pymupdf

    pdf = pymupdf.open()
    for _ in range(3):
        pdf.new_page()
    pages = list(pdf)     # 新建页面会让之前拿到的页面对象失效，所以先建完再统一取
    for i, page in enumerate(pages, 1):
        _pdf_header_footer(page, i, len(pages))

    # 第 1 页：通栏标题和摘要，下面是双栏正文
    p = pages[0]
    _pdf_text(p, (50, 60, 545, 95), f"{COMPANY} 2025 年度经营报告", size=22, align=1)
    _pdf_text(p, (50, 100, 545, 118), "战略规划部 · 2026 年 1 月", size=10, align=1)
    _pdf_text(p, (70, 130, 525, 200),
              "摘要：本报告总结公司 2025 年经营情况。全年营业收入 4,860 万元，同比增长 18.6%；"
              "华东区连续四个季度保持第一。报告第 1 章介绍背景与目标，第 2 章分析区域数据，"
              "第 3 章给出结论与下一年度行动计划。", size=10)
    p.draw_line((50, 210), (545, 210), color=(0.3, 0.3, 0.3), width=0.8)
    left, right = (50, 222, 290, 790), (305, 222, 545, 790)
    _pdf_text(p, (left[0], 222, left[2], 245), "1 背景与目标", size=15)
    _pdf_text(p, (left[0], 250, left[2], 400),
              "2025 年是公司实施“区域深耕”战略的第一年。公司在四个区域同步推进渠道下沉，"
              "重点提升华东、华南两个核心市场的客户留存率¹。全年新增签约客户 168 家，"
              "其中年度合同额超过 50 万元的大客户 23 家。", size=10.5)
    _pdf_text(p, (left[0], 405, left[2], 425), "1.1 年度目标", size=12.5)
    _pdf_text(p, (left[0], 430, left[2], 520),
              "● 营业收入达到 4,800 万元；\n● 客户留存率不低于 85%；\n● 新产品线收入占比达到 20%；\n"
              "● 研发投入占收入比重不低于 12%。", size=10.5)
    _pdf_text(p, (left[0], 525, left[2], 545), "1.2 组织调整", size=12.5)
    _pdf_text(p, (left[0], 550, left[2], 690),
              "上半年完成区域事业部改组，原销售中心拆分为四个区域事业部，各事业部对收入和回款"
              "直接负责。下半年设立客户成功部，统一负责续约与增购。", size=10.5)
    _pdf_text(p, (right[0], 222, right[2], 245), "1.3 市场环境", size=12.5)
    _pdf_text(p, (right[0], 250, right[2], 420),
              "行业整体增速放缓至 9% 左右，但企业数字化预算仍在增加。竞争对手普遍采取降价策略，"
              "公司坚持价值定价，通过交付质量和服务响应速度获得差异化优势。右栏这一段用来检验"
              "双栏版面的阅读顺序：正确的顺序应当先读完左栏，再读右栏。", size=10.5)
    _pdf_text(p, (right[0], 425, right[2], 445), "1.4 风险提示", size=12.5)
    _pdf_text(p, (right[0], 450, right[2], 600),
              "1. 大客户集中度偏高，前十大客户贡献 41% 的收入；\n2. 应收账款周转天数增至 68 天；\n"
              "3. 核心研发人员流失率为 7.5%，高于行业平均水平。", size=10.5)
    p.draw_line((50, 760), (150, 760), color=(0, 0, 0), width=0.5)
    _pdf_text(p, (50, 763, 545, 790), "¹ 客户留存率 = 年末仍在合作的客户数 / 年初客户数，数据来源：内部 CRM 系统。", size=8)

    # 第 2 页：合并单元格表格、公式、图表，最后一段跨到第 3 页
    p = pages[1]
    _pdf_text(p, (50, 60, 545, 82), "2 区域经营数据", size=15)
    _pdf_text(p, (50, 88, 545, 120), "表 1 列出了各区域分季度营收（单位：万元），上半年与下半年为合并表头。", size=10.5)
    cells = [(0, 0, 2, 1, "区域"), (0, 1, 1, 2, "上半年"), (0, 3, 1, 2, "下半年"), (0, 5, 2, 1, "全年合计")]
    cells += [(1, 1 + i, 1, 1, q) for i, q in enumerate(QUARTERS)]
    for r, region in enumerate(REGIONS, start=2):
        values = REVENUE[region]
        cells += [(r, 0, 1, 1, region)] + [(r, 1 + i, 1, 1, str(v)) for i, v in enumerate(values)]
        cells.append((r, 5, 1, 1, str(sum(values))))
    _pdf_text(p, (50, 122, 545, 138), "表 1 2025 年各区域季度营收", size=9.5, align=1)
    bottom = _pdf_table(p, 72, 142, [80, 70, 70, 70, 70, 90], 22, cells)
    _pdf_text(p, (50, bottom + 6, 545, bottom + 20), "注：数据未经审计。", size=8.5)

    _pdf_text(p, (50, bottom + 32, 545, bottom + 64), "增长率按复合增长率计算，见公式 (1)，其中 V0 为期初值、Vn 为期末值、n 为期数：", size=10.5)
    eq = formula(r"CAGR=\left(\frac{V_n}{V_0}\right)^{\frac{1}{n}}-1", 18)
    p.insert_image(pymupdf.Rect(200, bottom + 68, 395, bottom + 118), stream=eq)
    _pdf_text(p, (480, bottom + 85, 545, bottom + 100), "(1)", size=10.5, align=2)

    chart_top = bottom + 130
    p.insert_image(pymupdf.Rect(130, chart_top, 465, chart_top + 200), stream=bar_chart())
    _pdf_text(p, (50, chart_top + 204, 545, chart_top + 220), "图 1 2025 年各区域季度营收", size=9.5, align=1)
    _pdf_text(p, (50, chart_top + 230, 545, 800),
              "从图 1 可以看出，四个区域的营收均逐季上升，其中华北区第三季度增幅最大，环比增长 22.2%，"
              "主要得益于两个政务云项目在三季度完成验收。西南区基数较小，但全年增速达到 53.3%，是增长最快的区域，"
              "明年将继续加大在成都和重庆两地的投入，并计划在第二季度新设贵阳办事处以覆盖", size=10.5)

    # 第 3 页：接续上一页的句子，然后是结论、编号列表、代码块和参考文献
    p = pages[2]
    _pdf_text(p, (50, 60, 545, 100), "云南与贵州市场，预计可带来约 300 万元的新增收入。", size=10.5)
    _pdf_text(p, (50, 110, 545, 132), "3 结论与行动计划", size=15)
    _pdf_text(p, (50, 140, 545, 240),
              "(1) 2026 年收入目标定为 5,800 万元，同比增长 19.3%；\n"
              "(2) 将前十大客户收入占比降至 35% 以下；\n"
              "(3) 应收账款周转天数压降到 55 天以内；\n"
              "(4) 上线新一代数据分析平台，接口示例见下方代码。", size=10.5)
    p.draw_rect(pymupdf.Rect(50, 248, 545, 330), color=(0.7, 0.7, 0.7), fill=(0.96, 0.96, 0.96), width=0.5)
    _pdf_text(p, (58, 254, 540, 328),
              "def quarterly_growth(values):\n"
              "    \"\"\"Return quarter-over-quarter growth rates.\"\"\"\n"
              "    return [b / a - 1 for a, b in zip(values, values[1:])]\n"
              "\n"
              "print(quarterly_growth([320, 360, 410, 480]))", size=9, font="cour")
    _pdf_text(p, (50, 345, 545, 365), "参考文献", size=12.5)
    _pdf_text(p, (50, 370, 545, 440),
              "[1] 中国信息通信研究院. 中国数字经济发展研究报告[R]. 2025.\n"
              "[2] 星河科技财务部. 2025 年度财务快报[R]. 2026.", size=9.5)
    pdf.subset_fonts()
    pdf.save(path, garbage=4, deflate=True)


def pdf_scanned_contract(path: Path) -> None:
    """扫描件：页面是带噪点、轻微倾斜的图片，没有文字层；含表格、印章和手写体风格签名。"""
    import pymupdf
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rng = random.Random(42)
    width, height = 1240, 1754   # A4 @150dpi
    title_font = ImageFont.truetype(CJK_FONT, 44)
    font = ImageFont.truetype(CJK_FONT, 28)
    small = ImageFont.truetype(CJK_FONT, 24)

    def page_image(draw_content) -> bytes:
        img = Image.new("L", (width, height), 250)
        draw = ImageDraw.Draw(img)
        draw_content(draw)
        img = img.rotate(0.8, fillcolor=250, expand=False)
        noise = Image.effect_noise((width, height), 18)
        img = Image.blend(img, noise, 0.08).filter(ImageFilter.GaussianBlur(0.6))
        img = img.convert("RGB")
        if draw_content is page1:   # 红色印章画在最后，避免被灰度化
            d = ImageDraw.Draw(img)
            d.ellipse([820, 1380, 1080, 1640], outline=(200, 30, 30), width=8)
            d.text((950, 1510), f"{COMPANY}\n合同专用章", font=small, fill=(200, 30, 30), anchor="mm", align="center")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def page1(draw):
        draw.text((width / 2, 140), "技术服务合同", font=title_font, fill=20, anchor="mm")
        draw.text((120, 230), "合同编号：XH-2025-0386", font=font, fill=30)
        lines = [
            "甲方：星河科技有限公司",
            "乙方：云帆数据服务有限公司",
            "第一条  服务内容",
            "乙方为甲方提供数据平台运维服务，服务期限自 2025 年 3 月 1 日",
            "至 2026 年 2 月 28 日，共十二个月。",
            "第二条  服务费用及支付方式",
        ]
        y = 300
        for line in lines:
            draw.text((120, y), line, font=font, fill=30)
            y += 52
        cols = [120, 420, 680, 900, 1120]
        rows = ["付款期次", "付款比例", "金额（元）", "付款时间"], ["第一期", "30%", "54,000", "签约后 10 日"], \
            ["第二期", "40%", "72,000", "2025-09-01"], ["第三期", "30%", "54,000", "验收合格后"]
        for r, row in enumerate(rows):
            top = y + 10 + r * 56
            for c, text in enumerate(row):
                draw.rectangle([cols[c], top, cols[c + 1], top + 56], outline=40, width=2)
                draw.text(((cols[c] + cols[c + 1]) / 2, top + 28), text, font=small, fill=30, anchor="mm")
        y += 10 + len(rows) * 56 + 40
        for line in ["合同总金额：人民币壹拾捌万元整（¥180,000.00）。",
                     "第三条  违约责任",
                     "任何一方违约，应向守约方支付合同总金额 10% 的违约金。"]:
            draw.text((120, y), line, font=font, fill=30)
            y += 52
        draw.text((width / 2, height - 80), "第 1 页 共 2 页", font=small, fill=60, anchor="mm")

    def page2(draw):
        y = 140
        for line in ["第四条  保密条款",
                     "乙方对在服务过程中获知的甲方数据和商业信息负有保密义务，",
                     "保密期限为合同终止后三年。",
                     "第五条  争议解决",
                     "因本合同引起的争议，双方应协商解决；协商不成的，",
                     "提交甲方所在地人民法院诉讼解决。"]:
            draw.text((120, y), line, font=font, fill=30)
            y += 52
        y += 80
        draw.text((120, y), "甲方（盖章）：", font=font, fill=30)
        draw.text((680, y), "乙方（盖章）：", font=font, fill=30)
        draw.text((120, y + 70), "授权代表：张  明", font=font, fill=30)
        draw.text((680, y + 70), "授权代表：李  华", font=font, fill=30)
        draw.text((120, y + 140), "日期：2025 年 2 月 20 日", font=font, fill=30)
        draw.text((680, y + 140), "日期：2025 年 2 月 20 日", font=font, fill=30)
        draw.text((width / 2, height - 80), "第 2 页 共 2 页", font=small, fill=60, anchor="mm")

    pdf = pymupdf.open()
    for draw_content in (page1, page2):
        page = pdf.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=page_image(draw_content))
    pdf.subset_fonts()
    pdf.save(path, garbage=4, deflate=True)


def pdf_english_paper(path: Path) -> None:
    """英文论文：多个带编号的行间公式、三线表、算法框、参考文献。"""
    import pymupdf

    pdf = pymupdf.open()
    p = pdf.new_page()
    _pdf_text(p, (50, 50, 545, 80), "Hybrid Retrieval with Reciprocal Rank Fusion", size=18, font="tiro", align=1)
    _pdf_text(p, (50, 84, 545, 100), "A. Researcher, B. Engineer  |  Galaxy Tech Lab", size=10, font="tiro", align=1)
    _pdf_text(p, (70, 110, 525, 170),
              "Abstract. We study how dense and sparse retrievers can be combined. Reciprocal Rank Fusion (RRF) "
              "merges ranked lists without score calibration and improves nDCG@10 by 4.1 points on our benchmark.",
              size=9.5, font="tiro")
    _pdf_text(p, (50, 180, 545, 198), "1  Method", size=13, font="tibo")
    _pdf_text(p, (50, 202, 545, 230),
              "Given ranked lists from m retrievers, the fused score of document d is defined in Eq. (1):",
              size=10.5, font="tiro")
    p.insert_image(pymupdf.Rect(170, 232, 425, 282), stream=formula(r"s(d)=\sum_{i=1}^{m}\frac{w_i}{k+r_i(d)}", 18))
    _pdf_text(p, (500, 250, 545, 264), "(1)", size=10.5, font="tiro", align=2)
    _pdf_text(p, (50, 286, 545, 312),
              "BM25 scores a term t in document d with the saturation function in Eq. (2):", size=10.5, font="tiro")
    p.insert_image(pymupdf.Rect(130, 314, 465, 368), stream=formula(
        r"\mathrm{bm25}(t,d)=\mathrm{idf}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}", 15))
    _pdf_text(p, (500, 334, 545, 348), "(2)", size=10.5, font="tiro", align=2)

    _pdf_text(p, (50, 380, 545, 398), "2  Results", size=13, font="tibo")
    _pdf_text(p, (50, 402, 545, 418), "Table 1: Retrieval quality on the test set.", size=9.5, font="tiro", align=1)
    rows = [("Method", "Recall@10", "MRR@10", "nDCG@10"), ("BM25", "0.712", "0.534", "0.581"),
            ("Dense", "0.768", "0.571", "0.619"), ("Hybrid (RRF)", "0.823", "0.602", "0.660")]
    cols, top = [90, 230, 330, 430, 520], 424
    p.draw_line((cols[0], top), (cols[-1], top), width=1.2)           # 三线表：只有横线
    p.draw_line((cols[0], top + 20), (cols[-1], top + 20), width=0.6)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            _pdf_text(p, (cols[c], top + 4 + r * 20, cols[c + 1], top + 20 + r * 20), text, size=9.5,
                      font="tibo" if r == 0 else "tiro", align=0 if c == 0 else 1)
    p.draw_line((cols[0], top + 80), (cols[-1], top + 80), width=1.2)

    _pdf_text(p, (50, 520, 545, 538), "Algorithm 1  Reciprocal Rank Fusion", size=10.5, font="tibo")
    p.draw_rect(pymupdf.Rect(50, 540, 545, 640), color=(0, 0, 0), width=0.5)
    _pdf_text(p, (58, 546, 540, 638),
              "Input: ranked lists L1..Lm, weights w, constant k\n"
              "1: for each document d in union(L1..Lm) do\n"
              "2:     s(d) <- sum_i w_i / (k + rank_i(d))\n"
              "3: end for\n"
              "4: return documents sorted by s(d) descending", size=9, font="cour")
    _pdf_text(p, (50, 655, 545, 673), "References", size=13, font="tibo")
    _pdf_text(p, (50, 677, 545, 740),
              "[1] G. Cormack, C. Clarke, S. Buettcher. Reciprocal rank fusion outperforms Condorcet and individual "
              "rank learning methods. SIGIR 2009.\n[2] S. Robertson, H. Zaragoza. The probabilistic relevance "
              "framework: BM25 and beyond. 2009.", size=9, font="tiro")
    pdf.subset_fonts()
    pdf.save(path, garbage=4, deflate=True)


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------
def _add_hyperlink(paragraph, url: str, text: str) -> None:
    """python-docx 没有超链接 API，按 OOXML 结构手动插入。"""
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    rel_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    style = OxmlElement("w:rStyle")
    style.set(qn("w:val"), "Hyperlink")
    props.append(style)
    run.append(props)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def docx_manual(path: Path) -> None:
    """多级标题、混合格式文字、嵌套列表、合并单元格表格、图片与图注、公式图、页眉页脚、超链接、代码、分页。"""
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.header.paragraphs[0].text = f"{COMPANY} · 数据平台产品手册 v2.3"
    section.footer.paragraphs[0].text = "内部资料，请勿外传"

    doc.add_heading("数据平台产品手册", level=0)
    p = doc.add_paragraph("本手册适用于")
    p.add_run("数据平台 2.3 版本").bold = True
    p.add_run("，面向系统管理员和数据分析师。阅读前请确认已获得")
    run = p.add_run("管理员权限")
    run.italic = True
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    p.add_run("。更新日志见")
    _add_hyperlink(p, "https://example.com/changelog", "官方更新日志")
    p.add_run("。")

    doc.add_heading("1 安装部署", level=1)
    doc.add_heading("1.1 环境要求", level=2)
    doc.add_paragraph("部署前需要准备以下环境：")
    doc.add_paragraph("操作系统：Ubuntu 22.04 或 CentOS 8", style="List Bullet")
    doc.add_paragraph("内存不少于 16 GB", style="List Bullet")
    doc.add_paragraph("生产环境建议 32 GB", style="List Bullet 2")
    doc.add_paragraph("数据库：PostgreSQL 14 及以上", style="List Bullet")
    doc.add_heading("1.2 安装步骤", level=2)
    doc.add_paragraph("下载安装包并校验 SHA-256", style="List Number")
    doc.add_paragraph("解压后执行安装脚本：", style="List Number")
    code = doc.add_paragraph()
    code_run = code.add_run("tar -xzf dataplat-2.3.tar.gz\ncd dataplat && sudo ./install.sh --prefix /opt/dataplat")
    code_run.font.name = "Courier New"
    code_run.font.size = Pt(9)
    doc.add_paragraph("访问 http://<服务器IP>:8080 完成初始化", style="List Number")
    doc.add_heading("1.2.1 常见安装问题", level=3)
    doc.add_paragraph("若提示端口被占用，请修改 config.yaml 中的 server.port。", style="Intense Quote")

    doc.add_heading("2 计费说明", level=1)
    doc.add_paragraph("各版本价格如表 1 所示，标准版和企业版的年费合并显示在“订阅”列组下。")
    table = doc.add_table(rows=5, cols=4)
    table.style = "Table Grid"
    data = [["版本", "订阅", "", "备注"],
            ["", "月费（元）", "年费（元）", ""],
            ["基础版", "999", "9,990", "不含技术支持"],
            ["标准版", "2,999", "29,990", "工作日支持"],
            ["企业版", "面议", "面议", "7×24 小时支持"]]
    for r, row in enumerate(data):
        for c, text in enumerate(row):
            table.cell(r, c).text = text
    table.cell(0, 1).merge(table.cell(0, 2))          # 横向合并
    table.cell(0, 0).merge(table.cell(1, 0))          # 纵向合并
    table.cell(0, 3).merge(table.cell(1, 3))
    doc.add_paragraph("表 1 各版本价格", style="Caption")

    doc.add_heading("3 数据分析", level=1)
    doc.add_paragraph("平台内置的报表模块可以直接生成区域营收对比图，如图 1 所示。")
    doc.add_picture(io.BytesIO(bar_chart()), width=Inches(4.5))
    doc.add_paragraph("图 1 各区域季度营收对比", style="Caption")
    doc.add_paragraph("同比增长率按下式计算：")
    doc.add_picture(io.BytesIO(formula(r"g=\frac{x_t-x_{t-1}}{x_{t-1}}\times 100\%", 16)), width=Inches(2.2))

    doc.add_page_break()
    doc.add_heading("4 报销流程", level=1)
    doc.add_paragraph("使用平台费用模块报销时，单据按图 2 所示的四个环节流转，任一环节驳回都会退回申请人。")
    doc.add_picture(io.BytesIO(flow_diagram()), width=Inches(5.5))
    doc.add_paragraph("图 2 报销审批流程", style="Caption")
    doc.add_heading("附录 A 术语表", level=1)
    terms = doc.add_table(rows=3, cols=2)
    terms.style = "Table Grid"
    for r, (k, v) in enumerate([("术语", "说明"), ("RRF", "倒数排名融合，一种多路召回融合方法"),
                                 ("SLA", "服务等级协议，约定可用性与响应时间")]):
        terms.cell(r, 0).text, terms.cell(r, 1).text = k, v
    doc.save(path)


# ---------------------------------------------------------------------------
# PowerPoint
# ---------------------------------------------------------------------------
def pptx_review(path: Path) -> None:
    """标题页、多级项目符号、合并单元格表格、原生柱状图与饼图、图片、组合形状、双栏文本框、演讲者备注、纯图片页。"""
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "2025 年第四季度经营复盘"
    s.placeholders[1].text = f"{COMPANY} · 销售运营中心 · 2026 年 1 月"

    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "核心结论"
    body = s.placeholders[1].text_frame
    body.text = "全年营收 4,860 万元，完成目标的 101%"
    for text, level in [("华东区第四季度营收 480 万元，环比增长 17.1%", 1),
                        ("西南区全年增速 53.3%，为增长最快区域", 1),
                        ("回款周期变长，应收账款周转天数 68 天", 0),
                        ("需要重点关注前十大客户的续约", 1)]:
        para = body.add_paragraph()
        para.text, para.level = text, level
    s.notes_slide.notes_text_frame.text = "讲解要点：先说结论，再说风险，回款问题需要财务部配合。"

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "区域营收明细（万元）"
    rows, cols = 6, 6
    table = s.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.5), Inches(3.6)).table
    head = ["区域", "上半年", "", "下半年", "", "合计"]
    sub = ["", "Q1", "Q2", "Q3", "Q4", ""]
    for c in range(cols):
        table.cell(0, c).text, table.cell(1, c).text = head[c], sub[c]
    for r, region in enumerate(REGIONS, start=2):
        values = REVENUE[region]
        for c, text in enumerate([region, *map(str, values), str(sum(values))]):
            table.cell(r, c).text = text
    table.cell(0, 1).merge(table.cell(0, 2))
    table.cell(0, 3).merge(table.cell(0, 4))
    table.cell(0, 0).merge(table.cell(1, 0))
    table.cell(0, 5).merge(table.cell(1, 5))

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "季度营收对比（原生图表）"
    chart_data = CategoryChartData()
    chart_data.categories = QUARTERS
    for region in REGIONS:
        chart_data.add_series(region, REVENUE[region])
    chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(1.5),
                               Inches(7.5), Inches(5.3), chart_data).chart
    chart.has_legend, chart.legend.position = True, XL_LEGEND_POSITION.BOTTOM
    pie_data = CategoryChartData()
    pie_data.categories = REGIONS
    pie_data.add_series("全年占比", [sum(REVENUE[r]) for r in REGIONS])
    pie = s.shapes.add_chart(XL_CHART_TYPE.PIE, Inches(8.6), Inches(1.5), Inches(4.3), Inches(4.5), pie_data).chart
    pie.has_legend = True
    s.notes_slide.notes_text_frame.text = "柱状图和饼图都是 PowerPoint 原生图表，数据保存在文件内部。"

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "报销流程与改进措施"
    s.shapes.add_picture(io.BytesIO(flow_diagram()), Inches(0.8), Inches(1.6), width=Inches(11.5))
    cap = s.shapes.add_textbox(Inches(0.8), Inches(4.3), Inches(11.5), Inches(0.5)).text_frame
    cap.text = "图：报销审批流程（图片）"
    group = s.shapes.add_group_shape()
    for i, (title, text) in enumerate([("改进一", "审批节点从 5 个减少到 4 个"),
                                        ("改进二", "发票自动识别，减少手工录入")]):
        box = group.shapes.add_textbox(Inches(0.8 + i * 6), Inches(5.0), Inches(5.5), Inches(1.5)).text_frame
        box.text = title
        box.paragraphs[0].runs[0].font.bold = True
        box.add_paragraph().text = text

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "趋势图（仅图片，检验图片中的文字识别）"
    s.shapes.add_picture(io.BytesIO(line_chart()), Inches(2.5), Inches(1.5), height=Inches(5.5))

    s = prs.slides.add_slide(prs.slide_layouts[6])   # 空白版式：没有标题占位符
    box = s.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(2)).text_frame
    box.text = "谢谢！"
    box.paragraphs[0].runs[0].font.size = Pt(54)
    box.add_paragraph().text = "联系人：运营中心 王芳  分机 8021"
    prs.save(path)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def xlsx_sales(path: Path) -> None:
    """多工作表、合并标题行、公式、日期与百分比格式、空行空列、同表两块数据、原生图表、隐藏工作表。"""
    from datetime import date

    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    ws = wb.active
    ws.title = "订单明细"
    ws.merge_cells("A1:F1")
    ws["A1"] = "2025 年第四季度订单明细"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A1"].alignment = Alignment(horizontal="center")
    ws.append([])                                       # 第 2 行留空
    ws.append(["订单号", "下单日期", "区域", "产品", "数量", "单价（元）", "金额（元）", "折扣率"])
    orders = [("XH-1001", date(2025, 10, 8), "华东", "数据平台标准版", 3, 29990, 0.05),
              ("XH-1002", date(2025, 10, 21), "华南", "数据平台企业版", 1, 120000, 0.10),
              ("XH-1003", date(2025, 11, 3), "华北", "数据平台基础版", 10, 9990, 0.0),
              ("XH-1004", date(2025, 11, 17), "西南", "数据平台标准版", 2, 29990, 0.0),
              ("XH-1005", date(2025, 12, 9), "华东", "运维服务包", 5, 18000, 0.08)]
    for i, (no, d, region, product, qty, price, discount) in enumerate(orders, start=4):
        ws.append([no, d, region, product, qty, price, f"=E{i}*F{i}*(1-H{i})", discount])
        ws[f"B{i}"].number_format = "yyyy-mm-dd"
        ws[f"H{i}"].number_format = "0%"
    total_row = 4 + len(orders)
    ws[f"A{total_row}"] = "合计"
    ws[f"E{total_row}"] = f"=SUM(E4:E{total_row - 1})"
    ws[f"G{total_row}"] = f"=SUM(G4:G{total_row - 1})"
    # J 列起放第二块数据（中间隔一列空白），检验同一工作表里多个表格的识别
    ws["K3"], ws["L3"] = "产品", "库存（套）"
    for r, (product, stock) in enumerate([("数据平台基础版", 120), ("数据平台标准版", 45), ("运维服务包", 30)], start=4):
        ws[f"K{r}"], ws[f"L{r}"] = product, stock

    summary = wb.create_sheet("区域汇总")
    summary.append(["区域", *QUARTERS, "全年", "占比"])
    for r, region in enumerate(REGIONS, start=2):
        summary.append([region, *REVENUE[region], f"=SUM(B{r}:E{r})", f"=F{r}/SUM($F$2:$F$5)"])
        summary[f"G{r}"].number_format = "0.0%"
    chart = BarChart()
    chart.title, chart.y_axis.title = "各区域季度营收", "万元"
    chart.add_data(Reference(summary, min_col=2, min_row=1, max_col=5, max_row=5), titles_from_data=True)
    chart.set_categories(Reference(summary, min_col=1, min_row=2, max_row=5))
    summary.add_chart(chart, "I2")

    notes = wb.create_sheet("填表说明")
    notes["A1"] = "填表说明"
    notes["A1"].font = Font(bold=True)
    notes["A2"] = "1. 金额 = 数量 × 单价 ×（1 − 折扣率），由公式自动计算，请勿手工修改。"
    notes["A3"] = "2. 折扣率超过 10% 需要销售总监审批。"
    hidden = wb.create_sheet("临时计算")
    hidden["A1"] = "这是隐藏工作表中的内容"
    hidden.sheet_state = "hidden"
    wb.save(path)


# ---------------------------------------------------------------------------
# CSV / TSV
# ---------------------------------------------------------------------------
EMPLOYEES = [
    ["工号", "姓名", "部门", "入职日期", "备注"],
    ["E001", "王芳", "运营中心", "2019-03-01", "负责华东区运营"],
    ["E002", "李华", "研发部", "2020-07-15", "主导数据平台 2.3 版本，含\"实时计算\"模块"],
    ["E003", "张明", "销售部", "2021-11-02", "华南,西南两区大客户"],
    ["E004", "陈静", "财务部", "2022-05-20", "多行备注：\n第一行\n第二行"],
    ["E005", "赵磊", "客户成功部", "", "入职日期待补充"],
]


def csv_files(out: Path) -> None:
    with (out / "07_员工名单.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(EMPLOYEES)
    with (out / "08_员工名单_GBK编码.csv").open("w", encoding="gbk", newline="") as fh:
        csv.writer(fh).writerows(EMPLOYEES)
    with (out / "09_库存清单.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerows([["SKU", "产品", "仓库", "库存", "安全库存"],
                          ["SKU-01", "数据平台基础版", "上海仓", "120", "50"],
                          ["SKU-02", "数据平台标准版", "广州仓", "45", "60"],
                          ["SKU-03", "运维服务包", "北京仓", "30", "20"]])


# ---------------------------------------------------------------------------
# 网页 / 网页存档 / 电子书
# ---------------------------------------------------------------------------
def _article_html(img_src: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{COMPANY}发布数据平台 2.3 版本</title>
<style>body{{font-family:sans-serif}} table{{border-collapse:collapse}} td,th{{border:1px solid #999;padding:4px}}</style>
<script>console.log("页面脚本，不应出现在解析结果中");</script></head>
<body>
<nav><a href="/">首页</a> | <a href="/news">新闻中心</a> | <a href="/about">关于我们</a></nav>
<header><p>{COMPANY}官网 · 新闻中心</p></header>
<article>
<h1>{COMPANY}发布数据平台 2.3 版本</h1>
<p class="meta">发布时间：2026-01-15　作者：产品部</p>
<p>1 月 15 日，{COMPANY}正式发布数据平台 2.3 版本。新版本引入<strong>实时计算引擎</strong>，
查询延迟降低 <em>40%</em>，并支持公式 <code>E = mc²</code> 这类行内代码的展示。</p>
<h2>主要更新</h2>
<ol>
  <li>实时计算引擎
    <ul><li>支持 Kafka 与 Pulsar 数据源</li><li>窗口聚合延迟低于 1 秒</li></ul>
  </li>
  <li>新增 12 种图表类型</li>
  <li>权限模型升级为基于角色的访问控制（RBAC）</li>
</ol>
<h2>版本价格</h2>
<table>
  <tr><th rowspan="2">版本</th><th colspan="2">订阅价格（元）</th><th rowspan="2">技术支持</th></tr>
  <tr><th>月费</th><th>年费</th></tr>
  <tr><td>基础版</td><td>999</td><td>9,990</td><td>无</td></tr>
  <tr><td>标准版</td><td>2,999</td><td>29,990</td><td>工作日</td></tr>
  <tr><td>企业版</td><td colspan="2">面议</td><td>7×24 小时</td></tr>
</table>
<h3>升级命令</h3>
<pre><code>dataplat upgrade --to 2.3 --backup /data/backup</code></pre>
<blockquote>“新版本让我们的日报生成时间从 2 小时缩短到 10 分钟。” —— 某零售客户</blockquote>
<figure><img src="{img_src}" alt="各区域季度营收柱状图"><figcaption>图 1 各区域季度营收</figcaption></figure>
</article>
<aside>相关阅读：数据平台 2.2 版本发布说明</aside>
<footer>© 2026 {COMPANY}　京ICP备00000000号</footer>
</body></html>
"""


def html_page(path: Path) -> None:
    """导航/页脚噪声、多级标题、嵌套有序/无序列表、rowspan/colspan 表格、代码、引用、内嵌图片。"""
    data_uri = "data:image/png;base64," + base64.b64encode(bar_chart()).decode("ascii")
    path.write_text(_article_html(data_uri), encoding="utf-8")


def mhtml_archive(path: Path) -> None:
    """网页存档：HTML 与图片打包在 multipart/related 里，图片按 Content-Location 引用。"""
    base = "https://news.example.com/2026/dataplat-2-3"
    msg = MIMEMultipart("related", type="text/html")
    msg["Subject"] = f"{COMPANY}发布数据平台 2.3 版本"
    msg["Snapshot-Content-Location"] = f"{base}.html"
    html = MIMEText(_article_html(f"{base}/chart.png"), "html", "utf-8")
    html["Content-Location"] = f"{base}.html"
    msg.attach(html)
    image = MIMEImage(bar_chart(), "png")
    image["Content-Location"] = f"{base}/chart.png"
    msg.attach(image)
    path.write_bytes(msg.as_bytes())


def epub_book(path: Path) -> None:
    """EPUB 3：目录导航、两章正文（标题、列表、表格、图片）。"""
    chapter1 = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-CN"><head><title>第一章</title></head><body>
<h1>第一章 新员工入职</h1>
<p>欢迎加入{COMPANY}！本章介绍入职第一周需要完成的事项。</p>
<h2>1.1 第一周清单</h2>
<ul><li>领取工牌和电脑</li><li>完成信息安全培训</li><li>与导师进行首次一对一沟通</li></ul>
<h2>1.2 考勤制度</h2>
<p>公司实行弹性工作制，核心工作时间为 10:00 至 16:00。</p>
</body></html>"""
    chapter2 = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-CN"><head><title>第二章</title></head><body>
<h1>第二章 福利待遇</h1>
<table><tr><th>福利项目</th><th>标准</th></tr>
<tr><td>餐饮补贴</td><td>每月 600 元</td></tr><tr><td>年度体检</td><td>每年一次</td></tr>
<tr><td>带薪年假</td><td>入职满一年 10 天</td></tr></table>
<p>2025 年各区域营收如下图所示，绩效奖金与所在区域的业绩挂钩。</p>
<img src="images/chart.png" alt="各区域季度营收"/>
</body></html>"""
    nav = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>目录</title></head><body>
<nav epub:type="toc"><ol><li><a href="chapter1.xhtml">第一章 新员工入职</a></li>
<li><a href="chapter2.xhtml">第二章 福利待遇</a></li></ol></nav></body></html>"""
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="bookid">urn:uuid:4f6c1d0e-2b7a-4c55-9d9e-xinghe-handbook</dc:identifier>
<dc:title>{COMPANY}员工手册</dc:title><dc:language>zh-CN</dc:language>
<meta property="dcterms:modified">2026-01-01T00:00:00Z</meta></metadata>
<manifest>
<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
<item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
<item id="c2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
<item id="chart" href="images/chart.png" media-type="image/png"/>
</manifest>
<spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>"""
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)   # 必须是第一个且不压缩
        z.writestr("META-INF/container.xml", container, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/content.opf", opf, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/nav.xhtml", nav, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/chapter1.xhtml", chapter1, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/chapter2.xhtml", chapter2, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/images/chart.png", bar_chart(), compress_type=zipfile.ZIP_DEFLATED)


# ---------------------------------------------------------------------------
# 图片
# ---------------------------------------------------------------------------
def image_files(out: Path, report_pdf: Path) -> None:
    import pymupdf
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    # 文档截图：年度报告第 2 页（合并表头表格 + 公式 + 柱状图）
    with pymupdf.open(report_pdf) as pdf:
        pdf[1].get_pixmap(dpi=110).save(out / "20_文档截图_报告第2页.png")

    # 拍照风格的表格：倾斜、模糊、带噪点、有背景色，存为 JPG
    rng = random.Random(7)
    img = Image.new("RGB", (1000, 640), (236, 230, 214))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(CJK_FONT, 30)
    draw.text((500, 50), "办公用品领用登记表", font=font, fill=(25, 25, 25), anchor="mm")
    rows = [["日期", "领用人", "物品", "数量"], ["1月6日", "王芳", "签字笔", "10"],
            ["1月8日", "李华", "笔记本", "3"], ["1月9日", "张明", "打印纸", "2 箱"], ["1月12日", "陈静", "订书机", "1"]]
    xs = [80, 280, 480, 760, 920]
    for r, row in enumerate(rows):
        y = 100 + r * 95
        for c, text in enumerate(row):
            draw.rectangle([xs[c], y, xs[c + 1], y + 95], outline=(40, 40, 40), width=3)
            draw.text(((xs[c] + xs[c + 1]) / 2, y + 47), text, font=font, fill=(20, 20, 20), anchor="mm")
    img = img.rotate(-2.5, fillcolor=(200, 195, 180), expand=True).filter(ImageFilter.GaussianBlur(1.0))
    noise = Image.effect_noise(img.size, 30).convert("RGB")
    img = Image.blend(img, noise, 0.07)
    img.save(out / "21_拍照_领用登记表.jpg", quality=75)

    # 纯图表图片
    (out / "22_图表_季度趋势.png").write_bytes(line_chart())
    _ = rng   # 固定随机种子，保证重复生成结果一致


# ---------------------------------------------------------------------------
# LibreOffice 转换：旧版 Office、OpenDocument、RTF
# ---------------------------------------------------------------------------
def convert(src: Path, target: str, dest: Path) -> bool:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        print(f"  跳过 {dest.name}：未安装 LibreOffice")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        profile = Path(tmp) / "profile"
        result = subprocess.run([soffice, "--headless", "--norestore", f"-env:UserInstallation={profile.as_uri()}",
                                 "--convert-to", target, "--outdir", tmp, str(src)],
                                capture_output=True, text=True, timeout=300)
        produced = Path(tmp) / f"{src.stem}.{target.split(':')[0]}"
        if not produced.exists():
            # 只装了 libreoffice-core 时会报 "source file could not be loaded"，需要 writer/calc/impress 组件
            raise RuntimeError(f"LibreOffice 转换 {src.name} -> {target} 失败：{(result.stderr or result.stdout).strip()}")
        shutil.move(produced, dest)
    return True


def recalc_xlsx(path: Path) -> None:
    """openpyxl 只写公式不算结果；经 LibreOffice 重新保存一次，把公式计算值写进文件。"""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "src" / path.name
        copy.parent.mkdir()
        shutil.copy(path, copy)
        if convert(copy, "xlsx", Path(tmp) / "recalc.xlsx"):
            shutil.move(Path(tmp) / "recalc.xlsx", path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        if old.is_file():
            old.unlink()

    steps = [
        ("01_年度报告_双栏复杂版面.pdf", pdf_annual_report),
        ("02_扫描件_技术服务合同.pdf", pdf_scanned_contract),
        ("03_英文论文_公式与三线表.pdf", pdf_english_paper),
        ("04_产品手册.docx", docx_manual),
        ("05_季度复盘.pptx", pptx_review),
        ("06_销售数据.xlsx", xlsx_sales),
        ("10_新闻页面.html", html_page),
        ("11_新闻页面存档.mhtml", mhtml_archive),
        ("12_员工手册.epub", epub_book),
    ]
    for name, build in steps:
        build(OUT / name)
        print(f"  生成 {name}")
    recalc_xlsx(OUT / "06_销售数据.xlsx")
    csv_files(OUT)
    image_files(OUT, OUT / "01_年度报告_双栏复杂版面.pdf")

    conversions = [
        ("04_产品手册.docx", "doc", "13_产品手册_旧版.doc"),
        ("05_季度复盘.pptx", "ppt", "14_季度复盘_旧版.ppt"),
        ("06_销售数据.xlsx", "xls", "15_销售数据_旧版.xls"),
        ("04_产品手册.docx", "rtf", "16_产品手册.rtf"),
        ("04_产品手册.docx", "odt", "17_产品手册.odt"),
        ("05_季度复盘.pptx", "odp", "18_季度复盘.odp"),
        ("06_销售数据.xlsx", "ods", "19_销售数据.ods"),
    ]
    for src, target, name in conversions:
        if convert(OUT / src, target, OUT / name):
            print(f"  生成 {name}")

    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"\n共 {len(list(OUT.iterdir()))} 个文件，{total / 1024:.0f} KB → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
