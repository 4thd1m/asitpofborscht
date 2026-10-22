#!/usr/bin/env python3
"""Generate src/fonts/SITPDoubleStruck-Regular.woff2 from STIX Two Math.

KaTeX's \\mathbb (and so \\R, \\reals, \\N, ...) emits a plain ASCII letter in a
`.mathbb` span and draws it with KaTeX_AMS, the AMS msbm blackboard bold where
only the stem is doubled. To get STIX Two's outlined double-struck letters
instead, this builds a small webfont whose cmap points ASCII A-Z, a-z and 0-9 at
STIX Two Math's double-struck glyphs (U+1D538.., U+1D552.., U+1D7D8.., plus the
Letterlike holes such as U+211D for R), so a CSS font-family swap on `.mathbb`
is all theme/css/custom.css needs. Nothing in src/*.md changes.

Remapping and subsetting is a modification under the OFL, and "STIX" (an IEEE
trademark) and "TM Math" (a Reserved Font Name) may not name the result, so the
family is renamed "SITP Double Struck".

Usage (needs `pip install fonttools brotli`):
  gen-double-struck-font.py <STIXTwoMath-Regular.otf>
Get the .otf from CTAN (fonts/stix2-otf).
"""
import os
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

FAMILY = "SITP Double Struck"
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "fonts",
                   "SITPDoubleStruck-Regular.woff2")

# The capitals Unicode encoded before the Mathematical Alphanumeric block; their
# slots in U+1D538.. are reserved holes.
LETTERLIKE = {"C": 0x2102, "H": 0x210D, "N": 0x2115, "P": 0x2119,
              "Q": 0x211A, "R": 0x211D, "Z": 0x2124}


def main(src):
    font = TTFont(src)
    cmap = font.getBestCmap()
    remap = {}
    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        remap[ord(ch)] = cmap[LETTERLIKE.get(ch, 0x1D538 + i)]
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        remap[ord(ch)] = cmap[0x1D552 + i]
    for i, ch in enumerate("0123456789"):
        remap[ord(ch)] = cmap[0x1D7D8 + i]
    for table in font["cmap"].tables:
        if table.isUnicode():
            table.cmap = dict(remap)

    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = []
    opts.drop_tables += ["MATH"]
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    subsetter = subset.Subsetter(opts)
    subsetter.populate(unicodes=remap.keys())
    subsetter.subset(font)

    name = font["name"]
    for nid in (1, 2, 3, 4, 6, 16, 17, 18, 21, 22):
        name.removeNames(nameID=nid)
    name.setName(FAMILY, 1, 3, 1, 0x409)
    name.setName("Regular", 2, 3, 1, 0x409)
    name.setName(FAMILY + " Regular", 3, 3, 1, 0x409)
    name.setName(FAMILY + " Regular", 4, 3, 1, 0x409)
    name.setName(FAMILY.replace(" ", "") + "-Regular", 6, 3, 1, 0x409)
    font.save(OUT)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
