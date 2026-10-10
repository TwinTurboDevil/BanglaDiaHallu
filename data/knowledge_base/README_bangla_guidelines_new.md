# Knowledge base for BanglaDiaRAG

`data/knowledge_base/bangla_guidelines_new.md` is the curated Bangla text read by `src/Knowledge_Base_Construction.py`.

## What it is

- **83 distinct sections**, each organized as a self-contained knowledge chunk with a bold Bangla header and an explicit source tag.
- Passages were selected by the authors from national and international clinical guidelines, peer-reviewed literature, and public health sources, translated into formal Bangla using Gemini 3.1 Pro, and reviewed by two physicians for medical fidelity.
- Text represents an adaptation and translation of the underlying source materials rather than verbatim copying.
- **Size:** 10,496 words, 83 chunks.

## Relation to the study

- This is the primary knowledge-base file used for the main results reported in the paper.
- The file is used to construct and index the retrieval corpus for the BanglaDiaRAG system.

## Sources

| Source | Source-tagged sections | Terms of Reuse |
| :--- | :---: | :--- |
| **National Guideline on Diabetes Mellitus**, 1st ed., 2023 (DGHS, Bangladesh) | 73 | Translated and adapted for the research knowledge base; reuse remains subject to the applicable terms of the original source |
| **Guideline for Type 2 Diabetes Mellitus in Bangladesh**, 2003 | 3 | Adapted clinical reference material; reuse subject to the applicable terms of the original source |
| **National Protocol for Management of Diabetes and Hypertension**, 2018–19 | 3 | Government public-health guideline material; reuse subject to the applicable terms of the original source |
| **BADAS Guideline**, 2019 | 5 | Adapted for non-commercial research use; subject to the applicable terms of the original source |
| **CDC** (*Smoking and Diabetes*, *Road to Health Toolkit*, *Diabetes Plate Method*) | 4 | Most CDC material is public domain; attribution and non-endorsement requirements apply, with exceptions for third-party or otherwise restricted material |
| **WHO** (*Carbohydrate intake for adults and children*, 2023) | 1 | CC BY-NC-SA 3.0 IGO; non-commercial copying, translation, adaptation, and redistribution permitted with attribution and ShareAlike |
| **NHS** (*Starchy foods and carbohydrates*) | 3 | Open Government Licence (OGL) v3.0; copying and adaptation permitted subject to NHS terms, attribution, and applicable exclusions |
| **Amoah I, Cobbinah JC, Tawiah P, Lim JJ, Yeboah JA, Rush E. Glycaemic and satiety responses associated with the consumption of parboiled and braised rice: a systematic review and meta-analysis. Cogent Food \& Agriculture. 2025;11(1):2550070. doi:10.1080/23311932.2025.2550070** (*Cogent Food & Agriculture*, Taylor & Francis Online) | 1 | Open Access, CC BY 4.0; reuse, adaptation, and redistribution permitted with attribution |
| **NIDDK, NIH** (*Healthy Living with Diabetes*) | 1 | The majority of NIDDK website information is copyright-free and may be reproduced with attribution, subject to stated exceptions and conditions |
| **Meng J-M, Cao S-Y, Wei X-L, Gan R-Y, Wang Y-F, Cai S-X, et al. Effects and mechanisms of tea for the prevention and management of diabetes mellitus and diabetic complications: an updated review. Antioxidants. 2019;8(6):170. doi:10.3390/antiox8060170** (*Antioxidants*, MDPI) | 1 | Open Access (CC BY 4.0) scientific review adaptation |

**Note:** Source-tagged section counts overlap because some knowledge chunks cite more than one source. Therefore, the sum of the source-specific counts is greater than the total number of chunks. All 83 sections currently contain an explicit source tag.

Translated and adapted text remains subject to the applicable terms of its underlying source. Inclusion in this knowledge base does not imply official endorsement by DGHS, BADAS, CDC, WHO, NHS, NIDDK, or the authors or publishers of the cited scientific literature.

## License and Terms of Use

### Dataset License

The authors' original contributions to `data/knowledge_base/bangla_guidelines_new.md`, including **source selection, curation, original Bangla wording, structural chunking, and metadata organization**, are made available under a **[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)**, to the extent that these contributions are independently copyrightable and are not subject to third-party licensing requirements.

Under this license, the authors' licensed contributions may be:

- **Share:** copied and redistributed for research and educational purposes.
- **Adapt:** remixed, transformed, and built upon for academic benchmarks and RAG research.

The following conditions apply:

- **Attribution:** Appropriate credit must be given, a link to the license must be provided, and changes must be indicated.
- **NonCommercial:** The authors' CC BY-NC contributions may not be used for commercial purposes without prior permission.

### Third-Party Copyright and Licensing Notice

This knowledge base contains adapted and translated material derived from third-party clinical guidelines, government health resources, and scientific publications. The copyright and licensing status of each underlying source remains governed by the applicable rights and license of that source.

In particular:

- **WHO-derived material** remains subject to the **CC BY-NC-SA 3.0 IGO** requirements, including non-commercial use, attribution, and ShareAlike for adaptations. Any translation of WHO material should be clearly identified as an unofficial translation and should not imply WHO endorsement.
- **CC BY 4.0 scientific publications**, including Meng et al. (2019) and Amoah et al. (2025), permit reuse and adaptation with appropriate attribution.
- **CDC material** is generally public domain unless otherwise indicated, subject to CDC attribution and non-endorsement requirements.
- **NIDDK material** is generally copyright-free for most website content, subject to the exceptions and conditions specified by NIDDK.
- **NHS material** is released under the current Open Government Licence, including OGL v3.0 requirements stated in the NHS terms and conditions, and is subject to NHS-specific attribution, adaptation, refresh, third-party-content, and non-endorsement conditions.

The license applied to the authors' original contributions does **not override, replace, or expand the rights granted by any third-party source license**.

### Medical Disclaimer

> **Warning / Non-Clinical Notice:**
>
> This dataset is compiled strictly for computational linguistics, Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG) research purposes. It is **not** intended to be used as direct medical advice, clinical decision support, or a replacement for professional healthcare consultation.

## Rebuilding

To construct or re-index the vector knowledge base from the source markdown:

```bash
python src/Knowledge_Base_Construction.py \
  --kb-file data/knowledge_base/bangla_guidelines_new.md \
  --overwrite
```
