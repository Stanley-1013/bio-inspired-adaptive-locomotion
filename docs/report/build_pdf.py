#!/usr/bin/env python3
"""Render the generic English report (Final_Report.md) to a root-level PDF.

Pipeline: Final_Report.md -> (preprocess) -> pandoc -> tectonic -> Final_Report.pdf

The markdown's figure placeholders ("> [Figure N here] caption") carry no real
image; build_docx.py maps them to figures/ via FIGMAP. This script does the same
so the PDF actually embeds the figures, strips the custom cover/ToC block (pandoc
generates its own title + --toc), and compiles with a Unicode font (DejaVu) so the
Greek/math symbols (tau, omega, approx, sub/superscripts) render instead of
dropping out under the default Latin Modern.

Requires (conda-forge): pandoc, tectonic. Run from this directory:
    python build_pdf.py
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Final_Report.md")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "Final_Report.pdf"))  # repo root

# Same figure map as build_docx.py — keep in sync.
FIGMAP = {
    1: ["fig1_push.png"], 2: ["fig2_reward_curves.png"], 3: ["fig3_feasibility.png"],
    4: ["fig4a_ref_payload.png", "fig4b_nofatigue_payload.png"], 5: ["fig5_residual.png"],
    6: ["fig6_isaaclab_walk.png"], 7: ["fig7_isaaclab_crawl.png", "fig6_isaaclab_walk.png"],
}

TITLE = ("Reinforcement Learning for Bio-Inspired Torque-Based Quadruped "
         "Locomotion: A Cross-Simulator Reproduction and Analysis")
AUTHOR = ("Li Chuan-Han \\\\[2pt] {\\normalsize Department of Biomechatronics "
          "Engineering, National Taiwan University}")
DATE = "June 2026"

# Academic-aesthetic LaTeX preamble (compiled with xetex via tectonic):
# Palatino body, refined spacing, serif headings, journal-style captions.
HEADER = r"""
\usepackage{microtype}
\usepackage{setspace}
\setstretch{1.07}
\usepackage{ragged2e}
\setlength{\parindent}{1.3em}
\setlength{\emergencystretch}{3em}

% --- serif, scholarly section headings (KOMA) ---
\setkomafont{disposition}{\normalfont\rmfamily\bfseries}
\addtokomafont{section}{\Large}
\addtokomafont{subsection}{\large}
\addtokomafont{title}{\rmfamily\bfseries}

% --- journal-style figure/table captions ---
\usepackage[font=small,labelfont=bf,labelsep=period,width=.92\textwidth]{caption}

% --- lighter, professional table rules ---
\usepackage{booktabs}
\renewcommand{\arraystretch}{1.18}

% --- restrained academic link color ---
\usepackage{xcolor}
\definecolor{linknavy}{HTML}{1A3A6B}

% --- a hairline rule under the title block via section spacing ---
\RedeclareSectionCommand[beforeskip=1.4\baselineskip,afterskip=.6\baselineskip]{section}
\RedeclareSectionCommand[beforeskip=1.1\baselineskip,afterskip=.4\baselineskip]{subsection}
"""


# Three glyphs the Times-clone body font (Nimbus Roman) lacks; render them as
# LaTeX math instead. Applied only OUTSIDE code blocks (the mono font covers them
# there, and $...$ would show literally in verbatim).
_GLYPH_FIX = [("ḣ", r"$\dot{h}$"), ("vₓ", r"$v_{x}$"), ("ₓ", r"$_{x}$"),
              ("⁻⁶", r"$^{-6}$"), ("⁻", r"$^{-}$")]


def _fix_glyphs(s):
    for a, b in _GLYPH_FIX:
        s = s.replace(a, b)
    return s


def preprocess(md_path):
    lines = open(md_path, encoding="utf-8").read().split("\n")
    first_hr = next(i for i, l in enumerate(lines) if l.strip() == "---")
    body = lines[first_hr + 1:]
    fig_re = re.compile(r"^>?\s*\[\s*Figure\s*(\d+)\s*here\s*\]\s*(.*)$")
    out, i, skip_toc, in_code = [], 0, False, False
    while i < len(body):
        ln = body[i]; t = ln.strip()
        if t.startswith("```"):          # pass code fences through verbatim
            in_code = not in_code; out.append(ln); i += 1; continue
        if in_code:
            out.append(ln); i += 1; continue
        ln = _fix_glyphs(ln); t = ln.strip()
        if t == "## Contents":          # drop manual ToC; pandoc --toc replaces it
            skip_toc = True; i += 1; continue
        if skip_toc:
            if t == "---":
                skip_toc = False; i += 1; continue
            if t.startswith("## "):
                skip_toc = False
            else:
                i += 1; continue
        m = fig_re.match(t)
        if m:
            n, cap = int(m.group(1)), re.sub(r"\*\*", "", m.group(2)).strip()
            imgs = FIGMAP[n]
            if len(imgs) == 1:
                out.append(f"![{cap}](figures/{imgs[0]}){{width=78%}}")
            else:
                out.append(" ".join(f"![](figures/{im}){{width=48%}}" for im in imgs))
                out.append("")
                out.append(f"*Figure {n}. {cap}*")
            out.append(""); i += 1; continue
        out.append(ln); i += 1
    return "\n".join(out)


def main():
    pre = preprocess(SRC)
    tmp = os.path.join(HERE, ".Final_Report.pandoc.md")
    hdr = os.path.join(HERE, ".Final_Report.header.tex")
    open(tmp, "w", encoding="utf-8").write(pre)
    open(hdr, "w", encoding="utf-8").write(HEADER)
    cmd = [
        "pandoc", tmp, "-o", OUT, "--pdf-engine=tectonic",
        "--toc", "--toc-depth=2",
        "--include-in-header", hdr,
        "-V", f"title={TITLE}", "-V", f"author={AUTHOR}", "-V", f"date={DATE}",
        # KOMA-Script article: cleaner academic defaults than stock `article`.
        "-V", "documentclass=scrartcl", "-V", "classoption=11pt",
        # Nimbus Roman = high-quality Times clone (classic academic serif, IEEE-like);
        # Nimbus Sans headings; DejaVu mono for code (covers the Greek/math glyphs).
        "-V", "mainfont=Nimbus Roman",
        "-V", "sansfont=Nimbus Sans",
        "-V", "monofont=DejaVu Sans Mono", "-V", "monofontoptions=Scale=0.85",
        "-V", "geometry:a4paper", "-V", "geometry:margin=1in",
        "-V", "linestretch=1.07",
        "-V", "colorlinks=true",
        "-V", "linkcolor=linknavy", "-V", "urlcolor=linknavy", "-V", "citecolor=linknavy",
    ]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    os.remove(tmp); os.remove(hdr)
    if r.returncode != 0:
        sys.stderr.write(r.stderr); sys.exit(r.returncode)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
