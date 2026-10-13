# QIS101 presentation standard

This reference records Dave's preferred teaching and slide format from **Session 20 - Symplectic Methods.pptx**. When asked for a deck or addition "in QIS101 style," inspect the original deck and copy its real slide layouts and formatting. Use slide 20 for explanatory mathematics and slide 18 for a run/output figure. The original is the authority if a detail below is ambiguous.

## The observed format

| Element | Reference and rule |
| --- | --- |
| Canvas | Original PowerPoint 4:3 layout, 10 × 7.5 in. Keep its master, theme, margins, and slide numbers. |
| Explanatory title | Slide 20 title placeholder, 32 pt, centered in the same position. Preserve its theme font and direct formatting. |
| Explanatory body | Slide 20 body placeholder and paragraph objects: 24 pt text, inherited real PowerPoint bullet glyphs and hanging indentation, 12 pt paragraph spacing. Preserve natural multiline wrapping. Never type a `•` character into a text box. |
| Emphasis | Use the source's bold purple `#7030A0` selectively within a sentence, as for **Hauro Yoshida** on slide 20. Normal explanatory text is dark. |
| Mathematics | Display equations between the relevant explanatory bullets, centered within the body width. Use the source's native editable PowerPoint math format with Cambria Math at 24 pt. Use true typeset structures for stacked fractions, powers, and subscripts when the expression requires them. Inspect a rendered slide to confirm the viewer fallback matches. |
| Plot/output slide | Use slide 18's plot layout. Keep the title position, green `Run` emphasis, chart region, purple short labels beneath the chart, and bottom right page number. Edit **every** inherited label to describe the new output. |
| Page numbers | Preserve the slide-number placeholder and continue the sequence. |

## Teaching structure

Start from a concept students already learned; introduce the new physical equation; explain what every term does; show the derivation in small causal steps; then explain what a computed curve demonstrates and when an approximation is valid. A typical explanatory slide has three substantive sentence bullets and, where useful, one centered displayed equation. Slide 20 is the model for both content density and its explanatory progression. Avoid terse labels or summary bullets that leave the mechanism for the student to guess.

For a lab, refer to an actual `.ipynb` notebook's **executable code cells** and preceding Markdown cells. A `.py` script contains code sections, not Jupyter notebook cells. Cross-check cell numbers and plot descriptions against the notebook before referring to them on slides. Figures should come from the actual lab calculation and should have readable axis labels and a complete legend.

## Required visual check

Render both the original reference slide and each new slide at the same size. Compare title baseline, 24 pt body text, actual bullet indentation, paragraph spacing, equation centering/type, purple emphasis, chart placement, captions, and footer number. Fix mismatches before sharing. Check inherited text and chart labels, especially after copying a plot slide. Preserve the original slides intact when adding a section.

## Example application

The revised **Session 20 - Symplectic Methods QIS101 Format.pptx** adds slides 35–47 on the Krylov–Bogoliubov van der Pol lab using slide 20's teaching layout and slide 18's plotting layout. It accompanies `krylov_bogoliubov_van_der_pol_lab.ipynb` and `krylov_bogoliubov_van_der_pol_lab.md`.
