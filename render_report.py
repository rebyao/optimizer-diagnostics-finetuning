"""Render the short English Markdown report as a single A4 PDF; no experiments."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    source=ROOT/'deliverables/REPORT_EN.md'
    fig=plt.figure(figsize=(8.27,11.69))
    y=.95
    for block in source.read_text().strip().split('\n\n'):
        title=block.startswith('# ')
        heading=block.startswith('**') and block.endswith('**')
        text=block.replace('# ','',1) if title else block.replace('**','').strip('*')
        lines=textwrap.wrap(text,width=96 if not title else 65)
        size=14 if title else 10 if heading else 9.7
        height=.018 if title else .0165
        fig.text(.07,y,'\n'.join(lines),va='top',fontsize=size,
                 fontweight='bold' if title or heading else 'normal',linespacing=1.18)
        y-=len(lines)*height+.009
    assert y>.055, 'Report exceeds one page; shorten text'
    fig.text(.07,.035,'Sources: saved runs and CSV tables in deliverables/; figures provided separately.',fontsize=8,color='gray')
    fig.savefig(ROOT/'deliverables/REPORT_EN.pdf');plt.close(fig)
    print(f'One-page A4 report rendered; {len(source.read_text().split())} words.')
if __name__=='__main__': main()
