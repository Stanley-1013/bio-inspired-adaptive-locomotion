#!/usr/bin/env python3
"""把期末報告的 markdown 轉成符合範例格式的 .docx。

- 封面區塊(檔案開頭到第一個 '---')置中、放大。
- ## / ### 對應 Heading 1 / Heading 2。
- 表格(markdown pipe table)轉成 Word 表格。
- 程式碼區塊(``` 圍住)用等寬字型、淺底。
- 【此處插圖 N】... / [Figure N here] ... 占位段落 → 插入對應圖片(置中)+ 圖說。
- 中文用 Noto Sans CJK TC,西文/數字用 Times New Roman(對齊論文慣例)。

用法: python build_docx.py <input.md> <output.docx> [--figmap fig_map.json]
"""
import sys, re, json, os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# 占位符(N) → (圖檔, 圖說前綴去掉占位標記後的純文字由 md 提供)
# 對 fig4 這種「左右並排兩張」的,放一個 list
FIGMAP = {
    1: ["fig1_push.png"],
    2: ["fig2_reward_curves.png"],
    3: ["fig3_feasibility.png"],
    4: ["fig4a_ref_payload.png", "fig4b_nofatigue_payload.png"],
    5: ["fig5_residual.png"],
    6: ["fig6_isaaclab_walk.png"],
    7: ["fig7_isaaclab_crawl.png", "fig6_isaaclab_walk.png"],
}

def set_cjk_font(run, cjk="Noto Sans CJK TC", latin="Times New Roman", size=12, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), cjk)

def add_para(doc, text, size=12, bold=False, align=None, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    # 處理 **粗體** 標記
    parts = re.split(r'(\*\*.+?\*\*)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            r = p.add_run(part[2:-2]); set_cjk_font(r, size=size, bold=True)
        else:
            r = p.add_run(part); set_cjk_font(r, size=size, bold=bold)
    return p

def add_heading(doc, text, level):
    p = doc.add_paragraph()
    size = 16 if level == 1 else 13
    r = p.add_run(text); set_cjk_font(r, size=size, bold=True)
    p.space_after = Pt(6)
    return p

def add_image_block(doc, fignum, caption):
    figs = FIGMAP.get(fignum, [])
    for fn in figs:
        path = os.path.join(FIG_DIR, fn)
        if os.path.isfile(path):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            # 並排兩張時各縮小
            w = Inches(2.7) if len(figs) > 1 else Inches(4.5)
            run.add_picture(path, width=w)
    # 圖說
    cp = doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cp.add_run(caption); set_cjk_font(r, size=10, bold=False); r.font.italic = True

def parse_table(lines):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        rows.append(cells)
    # 移除分隔列(---)
    rows = [r for r in rows if not all(set(c) <= set('-: ') for c in r)]
    return rows

def add_table(doc, rows):
    if not rows: return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = 'Light Grid Accent 1'
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ''
            r = cell.paragraphs[0].add_run(row[j] if j < len(row) else '')
            set_cjk_font(r, size=10, bold=(i == 0))

def main():
    src, out = sys.argv[1], sys.argv[2]
    md = open(src, encoding='utf-8').read()
    lines = md.split('\n')
    doc = Document()
    # 預設樣式字型
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'; style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Noto Sans CJK TC')

    # 找封面結束(第一個 '---')
    first_hr = next(i for i, l in enumerate(lines) if l.strip() == '---')

    # --- 封面:置中 ---
    for l in lines[:first_hr]:
        t = l.strip()
        if not t:
            doc.add_paragraph(); continue
        # 標題行(中文長標題)放大
        big = (t.startswith('力矩控制') or t.startswith('Reproducing'))
        add_para(doc, t, size=15 if big else 12, bold=big, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()

    i = first_hr + 1
    fig_re = re.compile(r'^>?\s*[【\[]\s*(?:此處插圖|Figure)\s*(\d+)\s*(?:here)?\s*[】\]]\s*(.*)$')
    while i < len(lines):
        l = lines[i]; t = l.strip()
        if t == '---':
            i += 1; continue
        # 圖片占位
        m = fig_re.match(t)
        if m:
            add_image_block(doc, int(m.group(1)), re.sub(r'\*\*', '', m.group(2)))
            i += 1; continue
        # 標題
        if t.startswith('## '):
            add_heading(doc, t[3:].strip(), 1); i += 1; continue
        if t.startswith('### '):
            add_heading(doc, t[4:].strip(), 2); i += 1; continue
        # 程式碼區塊
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
        # 表格
        if t.startswith('|'):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                tbl.append(lines[i]); i += 1
            add_table(doc, parse_table(tbl)); continue
        # 清單
        if t.startswith('- '):
            add_para(doc, t[2:], style='List Bullet'); i += 1; continue
        if re.match(r'^\d+\.\s', t):
            add_para(doc, re.sub(r'^\d+\.\s', '', t), style='List Number'); i += 1; continue
        # 引用塊(非圖片)
        if t.startswith('> '):
            add_para(doc, t[2:], size=11); i += 1; continue
        # 空行
        if not t:
            i += 1; continue
        # 一般段落
        add_para(doc, t, size=12); i += 1

    doc.save(out)
    print(f"wrote {out}")

if __name__ == '__main__':
    main()
