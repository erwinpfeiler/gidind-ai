# Experimental peak lists

Four original CSVs are retained unchanged. Four derived CSVs end in `_without_specular.csv`. Use these derived lists for the planned GIDInd versus GIDInd-AI comparison without specular information.

Coordinates are in Å⁻¹. Filtering removes rows with |q_xy| ≤ 0.05 Å⁻¹ in the original physical coordinates. No scaling, q-range restriction, peak renumbering, rounding, or deduplication is applied. Retained source rows and all original columns are preserved verbatim. Rows at (0,0) are placeholders, not specular reflections. No specular information is synthesized for cases lacking it.

| Case | Original rows | Specular-free rows | Specular rows removed | Zero placeholders removed |
|---|---:|---:|---:|---:|
| pq | 75 | 74 | 0 | 1 |
| mapbbr3 | 15 | 12 | 2 | 1 |
| naib_s11 | 93 | 87 | 5 | 1 |
| cu_bod_mof | 10 | 10 | 0 | 0 |

PQ contains no measured specular peaks in the supplied list; its derived file only removes the zero placeholder. Cu-BOD contains neither specular peaks nor a zero placeholder, so its two files are byte-identical. MAPbBr3 and NAIB-S11 contain specular peaks that are removed in their derived versions. Original detector columns and peak identifiers in the perovskite CSVs remain unchanged; use the named q_xy and q_z columns rather than assuming their positions.

`manifest.json` records source and output SHA-256 hashes, removal counts, source provenance, and the exact filter. Both algorithms must receive the same retained physical peak coordinates; document any additional algorithm-specific scaling or q-range restrictions separately.
