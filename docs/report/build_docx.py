#!/usr/bin/env python3
"""把期末報告 markdown 轉成符合 NTU 範本的 .docx。

對齊範例(李宇皓報告 + NTU 論文範本)的關鍵格式:
- 封面左側「裝訂書脊」:直書(btLr)浮動文字方塊,內容為校系名 + 課程/標題/作者 + 年月。
  用 spine_template.xml(自範例抽出的原始 drawing XML)注入,只替換標題/作者/年月文字。
- 封面欄位:標楷體,系院 14pt、校名/課程/標題 16pt、作者/教授/日期 18pt,置中。
- 內文:中文標楷體、西文 Times New Roman、12pt。
- 章節標題用內建 Heading 1/2(可進導覽窗格與目次)。
- 中英雙摘要、表格、程式碼塊、圖片(置中+圖說)。
- 封面後不留多餘空白頁。

用法: python build_docx.py <input.md> <output.docx> [--spine "標題" "作者" "民國年" "月"]
"""
import sys, re, os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement, parse_xml

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
SPINE_TPL = os.path.join(HERE, "spine_template.xml")

KAI = "標楷體"          # 標楷體 / DFKai-SB
LATIN = "Times New Roman"

FIGMAP = {
    1: ["fig1_push.png"], 2: ["fig2_reward_curves.png"], 3: ["fig3_feasibility.png"],
    4: ["fig4a_ref_payload.png", "fig4b_nofatigue_payload.png"], 5: ["fig5_residual.png"],
    6: ["fig6_isaaclab_walk.png"], 7: ["fig7_isaaclab_crawl.png", "fig6_isaaclab_walk.png"],
}

# ---- 書脊文字(可由 CLI 覆寫) ----
SPINE = dict(title="力矩控制四足機器人仿生式運動控制的跨模擬引擎重現與實作",
             author="李傳漢", year="115", month="6")

def set_font(run, cjk=KAI, latin=LATIN, size=12, bold=False):
    run.font.size = Pt(size); run.font.bold = bold; run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.append(rf)
    rf.set(qn('w:eastAsia'), cjk)

def runs_with_bold(p, text, size=12, bold=False):
    for part in re.split(r'(\*\*.+?\*\*)', text):
        if not part: continue
        if part.startswith('**') and part.endswith('**'):
            set_font(p.add_run(part[2:-2]), size=size, bold=True)
        else:
            set_font(p.add_run(part), size=size, bold=bold)

def para(doc, text, size=12, bold=False, align=None, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None: p.alignment = align
    runs_with_bold(p, text, size=size, bold=bold)
    return p

def heading(doc, text, level, page_break=False):
    p = doc.add_heading(level=level)       # 用內建 Heading 樣式 → 進導覽窗格/目次
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if page_break:
        p.paragraph_format.page_break_before = True
    r = p.add_run(text); set_font(r, size=16 if level == 1 else 13, bold=True)
    return p

def image_block(doc, fignum, caption):
    figs = FIGMAP.get(fignum, [])
    if len(figs) > 1:
        # 並排:用單列表格放兩張,無框
        t = doc.add_table(rows=1, cols=len(figs))
        for k, fn in enumerate(figs):
            path = os.path.join(FIG_DIR, fn)
            cell = t.cell(0, k); cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if os.path.isfile(path):
                cp.add_run().add_picture(path, width=Inches(2.7))
    else:
        for fn in figs:
            path = os.path.join(FIG_DIR, fn)
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if os.path.isfile(path):
                p.add_run().add_picture(path, width=Inches(4.5))
    cp = doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cp.add_run(caption); set_font(r, size=10); r.font.italic = True

def parse_table(lines):
    rows = [[c.strip() for c in ln.strip().strip('|').split('|')] for ln in lines]
    return [r for r in rows if not all(set(c) <= set('-: ') for c in r)]

def add_table(doc, rows):
    if not rows: return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol); t.style = 'Table Grid'
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j); cell.text = ''
            r = cell.paragraphs[0].add_run(row[j] if j < len(row) else '')
            set_font(r, size=10, bold=(i == 0))

def insert_spine(doc, mode='zh'):
    """把直書書脊文字方塊注入封面第一段。mode='zh' 中文書脊 / 'en' 英文書脊。"""
    if not os.path.isfile(SPINE_TPL):
        print("  (警告:找不到 spine_template.xml,略過書脊)"); return None
    xml = open(SPINE_TPL, encoding='utf-8').read()
    if mode == 'en':
        # 英文書脊:校系名 + 課程 + 標題 + 作者(全英文)
        # 系名被拆成多個 run(生物機電工程學系( / 所 / 、 / 學位學程 / )),逐一處理
        xml = xml.replace('國立臺灣大學', 'National Taiwan University ')
        xml = xml.replace('生物機電工程學系(', 'Department of Biomechatronics Engineering')
        xml = xml.replace('學位學程', '')
        xml = xml.replace('人工智慧實作期末報告', SPINE.get('course_en', 'Final Report'))
        xml = xml.replace('  MCP ', '  ').replace('介紹及實作', SPINE['title'])
        xml = xml.replace('趙華杉', SPINE['author'])
        xml = xml.replace('撰', '')
        # 清掉系名殘餘的單字中文碎片(所、、、))——只在英文書脊
        for frag in ['所', '、', '）', '(', ')']:
            xml = xml.replace(f'<w:t>{frag}</w:t>', '<w:t></w:t>')
            xml = xml.replace(f'<w:t xml:space="preserve">{frag}</w:t>', '<w:t xml:space="preserve"></w:t>')
    else:
        # 中文書脊:替換範例文字。範例:「人工智慧實作期末報告  MCP 介紹及實作  趙華杉  撰」
        xml = xml.replace('  MCP ', '  ').replace('介紹及實作', SPINE['title'])
        xml = xml.replace('趙華杉', SPINE['author'])
    # 命名空間:解析時需補齊 root nsmap → 包一層帶完整宣告的 run
    NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
          'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
          'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
          'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
          'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
          'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
          'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
          'xmlns:v="urn:schemas-microsoft-com:vml"')
    run_xml = f'<w:r {NS}>{xml}</w:r>'
    try:
        run_el = parse_xml(run_xml)
        p = doc.add_paragraph()
        p._p.append(run_el)
        return p
    except Exception as e:
        print(f"  (警告:書脊 XML 注入失敗 {e};略過書脊,封面其餘正常)")
        return None

def main():
    src, out = sys.argv[1], sys.argv[2]
    # --spine-mode {zh,en,none};未指定則自動:有中文封面→zh,否則 none
    spine_mode = None
    if '--spine-mode' in sys.argv:
        spine_mode = sys.argv[sys.argv.index('--spine-mode') + 1]
    # --spine-title / --spine-course-en 覆寫書脊文字
    if '--spine-title' in sys.argv:
        SPINE['title'] = sys.argv[sys.argv.index('--spine-title') + 1]
    if '--spine-course-en' in sys.argv:
        SPINE['course_en'] = sys.argv[sys.argv.index('--spine-course-en') + 1]
    if '--spine-author' in sys.argv:
        SPINE['author'] = sys.argv[sys.argv.index('--spine-author') + 1]

    md = open(src, encoding='utf-8').read()
    lines = md.split('\n')
    doc = Document()
    st = doc.styles['Normal']
    st.font.name = LATIN; st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), KAI)

    first_hr = next(i for i, l in enumerate(lines) if l.strip() == '---')

    cover_txt = '\n'.join(lines[:first_hr])
    is_zh = any('一' <= c <= '鿿' for c in cover_txt)
    if spine_mode is None:
        spine_mode = 'zh' if is_zh else 'none'

    # --- 書脊 ---
    if spine_mode in ('zh', 'en'):
        insert_spine(doc, mode=spine_mode)

    # --- 封面:上中下三區垂直分布,填滿整頁 ---
    # 上區=校系/課程,中區=中英標題(較大),下區=作者/學號/教授/日期。
    # 用「區塊間的大段前間距」把三區分別推到頁面上/中/下,而非全擠頂部。
    cover_items = [l.strip() for l in lines[:first_hr] if l.strip()]
    def cover_role(t):
        if t.startswith('力矩控制') or t.startswith('以強化學習') or t.startswith('Reinforcement Learning for'):
            return 'title_zh'
        if t.startswith('Reproducing') or ('A Cross-Simulator' in t):
            return 'title_en'
        if t.startswith('李傳漢') or t.startswith('Li Chuan'):
            return 'author'
        if t.startswith('學號') or t.startswith('Student ID'):
            return 'id'
        if t.startswith('指導教授') or t.startswith('Teacher'):
            return 'advisor'
        if '2026' in t or '一一五' in t or '中華民國' in t:
            return 'date'
        return 'org'   # 系/院/校/課程
    def cover_size(role):
        return 16 if role in ('title_zh', 'title_en') else 14
    # 預估文字高度(含長行換行緩衝),再把剩餘空間分配到三個區塊間距,
    # 讓不同長度的封面(繁中 13 段 vs 通用 7 段)都填到約 84% 頁高、不偏上半。
    PAGE = 648.0; TARGET = 0.80 * PAGE          # 目標占用高度(留底部緩衝,避免長標題換行溢出)
    text_est = 0.0
    for t in cover_items:
        sz = cover_size(cover_role(t))
        text_est += sz * 1.2 + (sz * 1.2 if len(t) > 40 else 0)  # 長行 +1 行緩衝
    text_est += 6 * len(cover_items)             # 每段 space_after
    slack = max(0.0, TARGET - text_est)          # 可分配到三個大間距的餘裕
    # 三個大間距按比例分:頁頂 25% / 上→中 35% / 中→下 40%,各設下限
    gap_top   = max(30.0, slack * 0.25)
    gap_title = max(48.0, slack * 0.35)
    gap_auth  = max(54.0, slack * 0.40)
    prev_role = None
    for t in cover_items:
        role = cover_role(t)
        size = cover_size(role)
        bold = role in ('title_zh', 'title_en')
        p = para(doc, t, size=size, bold=bold, align=WD_ALIGN_PARAGRAPH.CENTER)
        pf = p.paragraph_format
        pf.space_after = Pt(6)
        if prev_role is None:
            pf.space_before = Pt(gap_top)
        elif role.startswith('title') and not (prev_role or '').startswith('title'):
            pf.space_before = Pt(gap_title)
        elif role == 'author' and (prev_role or '').startswith('title'):
            pf.space_before = Pt(gap_auth)
        else:
            pf.space_before = Pt(6)
        prev_role = role
    # 封面後分頁由「目次」H1 的 page_break_before 處理。

    i = first_hr + 1
    in_toc = False   # 是否在目次區(用於放鬆目次清單行距)
    fig_re = re.compile(r'^>?\s*[【\[]\s*(?:此處插圖|Figure)\s*(\d+)\s*(?:here)?\s*[】\]]\s*(.*)$')
    while i < len(lines):
        t = lines[i].strip()
        if t == '---':
            i += 1; continue
        m = fig_re.match(t)
        if m:
            image_block(doc, int(m.group(1)), re.sub(r'\*\*', '', m.group(2))); i += 1; continue
        if t.startswith('## '):
            htext = t[3:].strip()
            # 所有主區塊 H1(目次/摘要/Abstract/各章/參考文獻/開放原始碼)各自起新頁。
            hp = heading(doc, htext, 1, page_break=True)
            in_toc = htext in ('目次', 'Contents')
            if in_toc:
                hp.paragraph_format.space_after = Pt(24)  # 目次標題後留較大間距
            i += 1; continue
        if t.startswith('### '):
            heading(doc, t[4:].strip(), 2); i += 1; continue
        if t.startswith('```'):
            block = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                block.append(lines[i]); i += 1
            i += 1
            p = doc.add_paragraph()
            shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), 'F2F2F2')
            p._p.get_or_add_pPr().append(shd)
            r = p.add_run('\n'.join(block)); r.font.name = 'Consolas'; r.font.size = Pt(10)
            continue
        if t.startswith('|'):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                tbl.append(lines[i]); i += 1
            add_table(doc, parse_table(tbl)); continue
        if t.startswith('- '):
            para(doc, t[2:], style='List Bullet'); i += 1; continue
        if re.match(r'^\d+\.\s', t):
            para(doc, re.sub(r'^\d+\.\s', '', t), style='List Number'); i += 1; continue
        if t.startswith('> '):
            para(doc, t[2:], size=11); i += 1; continue
        if not t:
            i += 1; continue
        para(doc, t, size=12); i += 1

    doc.save(out)
    print(f"wrote {out}")

if __name__ == '__main__':
    main()
