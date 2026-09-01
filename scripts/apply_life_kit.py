"""One-shot transform: apply the life kit (FRONTEND_PRINCIPLES §8.7) to the
round-3 overview-console and load mocks. Idempotence guard: refuses to run
twice (checks for the manila token)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "mocks" / "round3"

# ---------- overview-console.html ----------
p = ROOT / "overview-console.html"
t = p.read_text(encoding="utf-8")
assert "--manila" not in t, "life kit already applied"

t = t.replace("""  --grid:#ECEEE9; --teal:#0D7066; --ok:#1E6E41; --warn:#92590F; --block:#A31D40;
  --chip:#EDEFEA;
}""", """  --grid:#ECEEE9; --teal:#0D7066; --ok:#1E6E41; --warn:#92590F; --block:#A31D40;
  --chip:#EDEFEA; --manila:#EDE5CE; --manila-ink:#6E5F38;
  --okbg:rgba(30,110,65,.09); --warnbg:rgba(146,89,15,.10); --blockbg:rgba(163,29,64,.08);
}""", 1)
t = t.replace("""    --grid:#1B201D; --teal:#43B3A6; --ok:#5CB582; --warn:#D6A25C; --block:#E27590;
    --chip:#202522;
  }
}""", """    --grid:#1B201D; --teal:#43B3A6; --ok:#5CB582; --warn:#D6A25C; --block:#E27590;
    --chip:#202522; --manila:#2B2718; --manila-ink:#CBB877;
    --okbg:rgba(92,181,130,.12); --warnbg:rgba(214,162,92,.13); --blockbg:rgba(226,117,144,.11);
  }
}""", 1)
t = t.replace("""    --grid:#1B201D; --teal:#43B3A6; --ok:#5CB582; --warn:#D6A25C; --block:#E27590;
    --chip:#202522;
}""", """    --grid:#1B201D; --teal:#43B3A6; --ok:#5CB582; --warn:#D6A25C; --block:#E27590;
    --chip:#202522; --manila:#2B2718; --manila-ink:#CBB877;
    --okbg:rgba(92,181,130,.12); --warnbg:rgba(214,162,92,.13); --blockbg:rgba(226,117,144,.11);
}""", 1)

t = t.replace("""h2{margin:0 0 8px;font:600 10.5px/1 "IBM Plex Sans",system-ui,sans-serif;
  letter-spacing:.15em;text-transform:uppercase;color:var(--dim);display:flex;gap:8px}
h2 .src{margin-left:auto;font-weight:400;text-transform:none;letter-spacing:0}""",
"""h2{margin:-10px -12px 10px;padding:7px 12px;font:600 10.5px/1 "IBM Plex Sans",system-ui,sans-serif;
  letter-spacing:.15em;text-transform:uppercase;color:var(--manila-ink);display:flex;gap:8px;
  background:var(--manila);border-bottom:1px solid var(--line);border-radius:4px 4px 0 0}
h2 .src{margin-left:auto;font-weight:400;text-transform:none;letter-spacing:0;opacity:.8}""")

t = t.replace(""".c-ok{color:var(--ok);border-color:var(--ok)} .c-warn{color:var(--warn);border-color:var(--warn)}
.c-block{color:var(--block);border-color:var(--block)}""",
""".c-ok{color:var(--ok);border-color:var(--ok);background:var(--okbg)}
.c-warn{color:var(--warn);border-color:var(--warn);background:var(--warnbg)}
.c-block{color:var(--block);border-color:var(--block);background:var(--blockbg)}""")

t = t.replace(""".figpanel{background:
  linear-gradient(var(--grid) 1px,transparent 1px),
  linear-gradient(90deg,var(--grid) 1px,transparent 1px),var(--panel);
  background-size:20px 20px,20px 20px,auto}""",
""".figpanel{position:relative;background:
  linear-gradient(var(--grid) 1px,transparent 1px),
  linear-gradient(90deg,var(--grid) 1px,transparent 1px),var(--panel);
  background-size:20px 20px,20px 20px,auto}
.figpanel::before,.figpanel::after{content:"";position:absolute;width:9px;height:9px;opacity:.7}
.figpanel::before{left:5px;bottom:5px;border-left:1.5px solid var(--dim);border-bottom:1.5px solid var(--dim)}
.figpanel::after{right:5px;bottom:5px;border-right:1.5px solid var(--dim);border-bottom:1.5px solid var(--dim)}
.ruler{height:9px;margin:-2px -12px 8px;opacity:.5;background:
  repeating-linear-gradient(90deg,var(--dim) 0 1px,transparent 1px 6px),
  repeating-linear-gradient(90deg,var(--dim) 0 1px,transparent 1px 30px);
  background-size:auto 4px,auto 9px;background-repeat:repeat-x;background-position:bottom left}
.portrait .ch-spokes{opacity:.5}
.portrait g,.portrait .ch-ring{transition:opacity .15s}
.portrait[data-hl] .ch-spokes,.portrait[data-hl] .ch-hatch,
.portrait[data-hl] .ch-dots,.portrait[data-hl] .ch-ring{opacity:.10}
.portrait[data-hl="ch-spokes"] .ch-spokes{opacity:1}
.portrait[data-hl="ch-hatch"] .ch-hatch{opacity:1}
.portrait[data-hl="ch-dots"] .ch-dots{opacity:1}
.portrait[data-hl="ch-ring"] .ch-ring{opacity:1}""")

t = t.replace(""".g:first-of-type{border-top:0}""",
""".g:first-of-type{border-top:0}
.g[data-ch]{cursor:default}
.g[data-ch]:hover{background:var(--chip);margin:0 -12px;padding-left:12px;padding-right:12px}""")
t = t.replace(""".btn:hover{background:var(--chip)}""",
""".btn:hover{background:var(--chip)} .btn:active{transform:translateY(1px)}""")
t = t.replace(""".room:hover{border-color:var(--teal)}""",
""".room:hover{border-color:var(--teal);background:var(--chip)}""")

BRAND = ('<div class="subject">\n    '
         '<span style="font:600 13px/1 \'IBM Plex Sans\',system-ui,sans-serif;'
         'letter-spacing:.1em">MRI-<b style="color:var(--teal)">LAB</b></span>\n    '
         '<span style="color:var(--dim)">/</span>\n    <svg class="face" ')
t = t.replace('<div class="subject">\n    <svg class="face" ', BRAND)

t = t.replace('<h2>State portrait <span class="src">canon v1 · dots frozen = coordinates annealed</span></h2>',
'<h2>State portrait <span class="src">canon v1 · dots frozen = coordinates annealed</span></h2>\n'
'      <div class="ruler" aria-hidden="true"></div>')

t = t.replace('<circle cx="170" cy="170" r="160" fill="none" stroke="var(--dim)" stroke-dasharray="5 4"/>',
              '<circle class="ch-ring" cx="170" cy="170" r="160" fill="none" stroke="var(--dim)" stroke-dasharray="5 4"/>')
t = t.replace('<g stroke="var(--teal)" stroke-width="1" opacity=".5">',
              '<g class="ch-spokes" stroke="var(--teal)" stroke-width="1">')
t = t.replace('<g stroke="var(--warn)" stroke-width="1.1">',
              '<g class="ch-hatch" stroke="var(--warn)" stroke-width="1.1">')
t = t.replace('<g fill="var(--ink)">', '<g class="ch-dots" fill="var(--ink)">')
t = t.replace('</svg>\n      <div class="figcap">',
'<g stroke="var(--dim)" opacity=".8"><line x1="163" y1="170" x2="177" y2="170"/>'
'<line x1="170" y1="163" x2="170" y2="177"/></g></svg>\n      <div class="figcap">')

t = t.replace('<div class="g"><span class="sym" style="color:var(--ink)">◦</span>',
              '<div class="g" data-ch="ch-dots"><span class="sym" style="color:var(--ink)">◦</span>')
t = t.replace('<div class="g"><span class="sym" style="color:var(--teal)">╱</span>',
              '<div class="g" data-ch="ch-spokes"><span class="sym" style="color:var(--teal)">╱</span>')
t = t.replace('<div class="g"><span class="sym hot">⌇</span>',
              '<div class="g" data-ch="ch-hatch"><span class="sym hot">⌇</span>')
t = t.replace('<div class="g"><span class="sym" style="color:var(--dim)">◌</span>',
              '<div class="g" data-ch="ch-ring"><span class="sym" style="color:var(--dim)">◌</span>')

t = t.replace("""    <span style="margin-left:auto">page ≈ 27 KB · 0 JS</span>
  </footer>
</div>""",
"""    <span style="margin-left:auto">page ≈ 28 KB · 0.4 KB JS (channel highlight)</span>
  </footer>
</div>
<script>
const P=document.querySelector('.portrait');
document.querySelectorAll('.g[data-ch]').forEach(function(r){
  r.addEventListener('pointerenter',function(){P.setAttribute('data-hl',r.dataset.ch)});
  r.addEventListener('pointerleave',function(){P.removeAttribute('data-hl')});
});
</script>""")
p.write_text(t, encoding="utf-8", newline="\n")
print("overview-console: life kit applied,", len(t)//1024, "KB")

# ---------- load.html: light touch ----------
p2 = ROOT / "load.html"
t2 = p2.read_text(encoding="utf-8")
t2 = t2.replace(""".c-ok{color:var(--ok);border-color:var(--ok)} .c-warn{color:var(--warn);border-color:var(--warn)}""",
""".c-ok{color:var(--ok);border-color:var(--ok);background:rgba(30,110,65,.09)}
.c-warn{color:var(--warn);border-color:var(--warn);background:rgba(146,89,15,.10)}""")
t2 = t2.replace(""".btn:hover{background:var(--chip)} .btn.primary:hover{opacity:.92}""",
""".btn:hover{background:var(--chip)} .btn.primary:hover{opacity:.92}
.btn:active{transform:translateY(1px)}""")
t2 = t2.replace(""".rcard:hover{border-color:var(--teal)}""",
""".rcard:hover{border-color:var(--teal);background:var(--chip)}""")
p2.write_text(t2, encoding="utf-8", newline="\n")
print("load: light kit applied")
