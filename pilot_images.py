"""
pilot_images.py  [n_pairs=24]

Feasibility pilot for recovering gene mentions we currently miss, on a sample of "supplement present
but 0 mentions" unlock candidates. Two NON-LLM sources (no tokens):
  (1) CAPTIONS/LEGENDS  -- all <caption>/<label>/<title> text in fullTextXML.
  (2) OCR of IMAGES     -- OCR EVERY substantive image (bounded), and LOG per-image details
                           (dims, bytes, ocr chars, gene hit) so we can derive a minimal filter.

Outputs pilot_results.csv (per gene-pair) + image_log.csv (per image) + a printed summary that
characterises which images actually yield gene hits. OCR: Tesseract (pytesseract) if available.
"""
import io
import os
import re
import sys
import zipfile
import shutil
import random
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '.')
from pipeline import supplementary_helpers as S
from pipeline.vpdb_helpers import get_gene_synonyms

UNLOCK = 'curated_data/all_PDs_with_PMID_2026_preprocessed_with_supplementary.csv'
OUT = 'out/supplementary_eval/pilot_images'
CAPS = dict(S.DEFAULT_CAPS)
IMG_EXT = {'.gif', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
MIN_DIM = 300            # skip thumbnails / icons
MIN_BYTES = 8 * 1024     # skip tiny images (often low-res gif duplicates)
MAX_PIXELS = 60_000_000  # skip enormous rasters
MAX_IMG_PER_PAPER = 30
random.seed(7)


def get_ocr():
    try:
        import pytesseract
        cands = [shutil.which('tesseract'), r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                 os.path.expanduser(r'~\Tesseract-OCR\tesseract.exe')]
        p = next((c for c in cands if c and os.path.exists(c)), None)
        if p:
            pytesseract.pytesseract.tesseract_cmd = p
        pytesseract.get_tesseract_version()
        def _ocr(img):
            try:
                return pytesseract.image_to_string(img, timeout=20)
            except Exception:
                return ""
        return 'tesseract', _ocr
    except Exception:
        import numpy as np
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return 'easyocr', lambda img: '\n'.join(reader.readtext(np.array(img.convert('RGB')), detail=0))


def norm(s):
    return re.sub(r'[^A-Z0-9]', '', str(s).upper().replace('O', '0'))


def captions_text(pmcid):
    try:
        r = S._http_get(f"{S.EPMC_BASE}/{pmcid}/fullTextXML")
        from lxml import etree
        root = etree.fromstring(r.content)
        out = []
        for el in root.iter():
            if isinstance(el.tag, str) and etree.QName(el).localname in ('caption', 'label', 'title'):
                out.append(" ".join("".join(el.itertext()).split()))
        return "\n".join(out)
    except Exception:
        return ""


def ocr_all_images(zf, ocr_fn):
    """OCR every substantive image (bounded). Returns [(name, w, h, kb, text)]."""
    from PIL import Image
    res, count = [], 0
    def walk(z, depth):
        nonlocal count
        for info in z.infolist():
            if count >= MAX_IMG_PER_PAPER or info.is_dir() or info.file_size > CAPS['max_file_bytes']:
                continue
            name = os.path.basename(info.filename)
            ext = os.path.splitext(name)[1].lower()
            try:
                data = z.read(info)
            except Exception:
                continue
            if ext == '.zip' and depth == 0:
                try:
                    walk(zipfile.ZipFile(io.BytesIO(data)), 1)
                except Exception:
                    pass
                continue
            if ext not in IMG_EXT or len(data) < MIN_BYTES:
                continue
            try:
                img = Image.open(io.BytesIO(data))
                w, h = img.size
                if min(w, h) < MIN_DIM or w * h > MAX_PIXELS:
                    continue
                txt = ocr_fn(img)
            except Exception:
                continue
            res.append((name, w, h, round(len(data) / 1024), txt))
            count += 1
    walk(zf, 0)
    return res


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    os.makedirs(OUT, exist_ok=True)
    eng, ocr_fn = get_ocr()
    print(f"OCR engine: {eng}", flush=True)

    u = pd.read_csv(UNLOCK, dtype=str)
    _b = lambda s: s.astype(str).str.upper() == 'TRUE'
    cand = u[_b(u['paper_available']) & ~_b(u['alias_in_text']) & _b(u['suppl_available'])
             & (pd.to_numeric(u['suppl_mentions'], errors='coerce').fillna(0) == 0)]
    papers = list(cand.groupby('pmid_CLEAN'))
    random.shuffle(papers)

    selected, cnt = [], 0
    for pmid, grp in papers:
        selected.append((pmid, grp)); cnt += len(grp)
        if cnt >= n:
            break
    print(f"selected {len(selected)} papers (~{cnt} pairs), 8 threads", flush=True)

    rows, imglog, done = [], [], [0]
    lock = threading.Lock()

    def work(pmid, grp):
        pmcid = S._normalize_pmcid(pmid)
        if not pmcid:
            return
        zf = S.fetch_supplementary_zip(pmcid, caps=CAPS)
        if zf is None:
            return
        cap = norm(captions_text(pmcid))
        imgs = ocr_all_images(zf, ocr_fn)
        img_norm = [(nm, w, h, kb, norm(txt), len(txt)) for nm, w, h, kb, txt in imgs]
        genes, lrows, limg = [], [], []
        for _, r in grp.iterrows():
            gene, db = r['Gene ID'], r['Database']
            try:
                aliases = get_gene_synonyms(gene, "", db)
            except Exception:
                aliases = []
            terms = [norm(t) for t in [gene] + list(aliases) if len(str(t)) >= 4]
            genes.append((gene, terms))
            in_cap = any(t in cap for t in terms)
            hit_imgs = [nm for nm, w, h, kb, t, tl in img_norm if any(term in t for term in terms)]
            lrows.append({'pmid': pmid, 'gene': gene, 'db': db, 'in_caption': in_cap,
                          'in_ocr_image': bool(hit_imgs), 'ocr_files': ";".join(hit_imgs), 'n_images_ocrd': len(imgs)})
        for nm, w, h, kb, t, tl in img_norm:
            hg = [g for g, terms in genes if any(term in t for term in terms)]
            limg.append({'pmid': pmid, 'image': nm, 'w': w, 'h': h, 'kb': kb, 'ocr_chars': tl,
                         'gene_hit': bool(hg), 'genes': ";".join(hg)})
        with lock:
            rows.extend(lrows); imglog.extend(limg); done[0] += 1
            pd.DataFrame(rows).to_csv(os.path.join(OUT, 'pilot_results.csv'), index=False)
            pd.DataFrame(imglog).to_csv(os.path.join(OUT, 'image_log.csv'), index=False)
            if done[0] % 10 == 0:
                print(f"  {done[0]}/{len(selected)} papers | {len(rows)} pairs | {len(imglog)} images", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in [ex.submit(work, p, g) for p, g in selected]:
            f.result()

    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'pilot_results.csv'), index=False)
    ilog = pd.DataFrame(imglog); ilog.to_csv(os.path.join(OUT, 'image_log.csv'), index=False)
    ntot = len(df)
    print("\n" + "=" * 62)
    print(f"PILOT ({eng}): {ntot} 'supplement-present-no-mention' pairs; {len(ilog)} images OCR'd")
    print(f"  via CAPTIONS/legends : {int(df['in_caption'].sum())} ({100*df['in_caption'].mean():.0f}%)")
    print(f"  via OCR (any image)  : {int(df['in_ocr_image'].sum())} ({100*df['in_ocr_image'].mean():.0f}%)")
    print(f"  via EITHER           : {int((df['in_caption']|df['in_ocr_image']).sum())} "
          f"({100*(df['in_caption']|df['in_ocr_image']).mean():.0f}%)")
    if len(ilog):
        hit = ilog[ilog['gene_hit']]
        print(f"\n  images with a gene hit: {len(hit)}/{len(ilog)}")
        if len(hit):
            print(f"    hit-image dims (WxH) median: {int(hit.w.median())}x{int(hit.h.median())}; "
                  f"kb median {int(hit.kb.median())}; ocr_chars median {int(hit.ocr_chars.median())}")
            print(f"    non-hit-image ocr_chars median: {int(ilog[~ilog.gene_hit].ocr_chars.median())}")
            print("    example hit images:", list(hit['image'].head(6)))
    print("saved ->", OUT)


if __name__ == '__main__':
    main()
