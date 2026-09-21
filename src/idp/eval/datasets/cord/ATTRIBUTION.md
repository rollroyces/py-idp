# Attribution — CORD eval dataset

The 30 receipt fixtures in `docs/` and the schema shape they follow
are derived from the **CORD: Consolidated Receipt Dataset for
Post-OCR Parsing** (Park et al., 2019), released by Clova AI / NAVER.

## Verbatim citation (BibTeX)

```bibtex
@inproceedings{cord2020,
  title     = {CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author    = {Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop
               and Lim, Jaeheung and Lee, Sangwoo and Kwon, Hyungho
               and Kim, Minsoo and Lee, Hwalsuk},
  booktitle = {Document Intelligence Workshop at Neural Information
               Processing Systems},
  year      = {2019},
  note      = {Workshop paper, NeurIPS 2019},
  url       = {https://github.com/clovaai/cord}
}
```

## Related publications

```bibtex
@inproceedings{cord2020main,
  title     = {A Single Multi-Task Deep Neural Network for Post-OCR
               Text Recognition},
  author    = {Hwang, Wonseok and Kim, Seonghyeon and Yim, Jinyeong
               and Seo, Minjoon and Park, Seunghyun and Park, Sungrae
               and Lee, Junyeop and Lee, Bado and Lee, Hwalsuk},
  booktitle = {Document Intelligence Workshop at Neural Information
               Processing Systems},
  year      = {2019}
}

@article{hwang2019post,
  title     = {Post-OCR parsing: building simple and robust parser via
               BIO tagging},
  author    = {Hwang, Wonseok and Kim, Seonghyeon and Yim, Jinyeong
               and Seo, Minjoon and Park, Seunghyun and Park, Sungrae
               and Lee, Junyeop and Lee, Bado and Lee, Hwalsuk},
  booktitle = {Document Intelligence Workshop at Neural Information
               Processing Systems},
  year      = {2019}
}
```

## Original source

- Repository: <https://github.com/clovaai/cord>
- Paper: <https://openreview.net/pdf?id=SJl3z659UH>
- License: Creative Commons Attribution 4.0 International
  (CC-BY-4.0) — see `LICENSE.txt` for the verbatim text.

## What py-idp ships

The 30 receipt text files under `docs/` are **synthetic fixtures**
written to match the CORD receipt shape (merchant header, line
items with `qty × unit_price = total`, subtotal/tax/tip/total,
payment method, last-4 card). They are not copies of any specific
receipt in the public CORD release — they were authored from scratch
to exercise the `Receipt` schema in `src/idp/core/schemas.py` for
the v0.4 baseline eval (ROADMAP_v0.4.md, item C2).

Because the fixtures reproduce the CORD *schema* and the
CC-BY-4.0 license is the upstream license declared in the CORD
README, the CC-BY-4.0 text is bundled in `LICENSE.txt` and this
attribution is required whenever these files (or the eval pipeline
that consumes them) are redistributed.

## What py-idp does NOT ship

- The original ~1000 CORD receipt images and their per-receipt OCR
  ground truth (`clovaai/cord`'s `dataset/` directory).
- The original CORD JSON annotations.
- Any line items, merchant names, or amounts copied from the public
  CORD release.

If you need those for a real benchmark, download them from
<https://github.com/clovaai/cord> and write your own converter from
CORD's per-image JSON shape into our `manifest.json` schema.