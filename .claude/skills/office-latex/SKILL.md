---
name: office-latex
description: Produce Office-compatible LaTeX for the Microsoft 365 Equation Editor (PowerPoint or Word, Insert -> Equation -> LaTeX input -> Convert to Math). Use whenever asked for "PowerPoint LaTeX", "Microsoft LaTeX", "Word equation", or "Equation Editor LaTeX", and whenever writing Dirac bra-ket notation, kets, bras, inner products, outer products, or tensor products that will be pasted into a slide or document. The Office build-up engine has stricter delimiter rules than a real LaTeX compiler and supports no packages, so ordinary LaTeX often fails there.
---

# LaTeX for PowerPoint / Word Equation Editor

When I ask for LaTeX to paste into the **Microsoft 365 Equation Editor**
(PowerPoint or Word: Insert -> Equation -> type LaTeX -> Convert to Math /
"build up"), produce **Office-compatible** LaTeX, not general LaTeX. The
Office build-up engine has stricter delimiter rules than a normal LaTeX
compiler and supports no packages at all, so expressions that render fine in
a real LaTeX compiler can "fail miserably" here.

Assume the equation is going into the Equation Editor in **LaTeX input
mode**, and return the raw source in a code block so it can be copied
directly.

## Core rule: delimiters must be balanced by count

Office pairs every opening delimiter (`(`, `[`, `|`, `\langle`, `\lfloor`, ...)
with a matching closer, then builds one auto-sizing bracket object between them.
An **unmatched opener escapes its group** and swallows surrounding content
(e.g. it eats across a fraction bar), producing a mangled result.

- Bad: `\frac{\lvert 1}{2}` - lone `\lvert` has no closer; the bar escapes the
  numerator and wraps the whole fraction.
- Good: `\frac{|1|}{2}` or `\frac{\left|1\right|}{2}` - balanced.

Office does **not** require the two sides to be the *same glyph* - only that
they form one matched `\left ... \right` pair. That is what makes
mixed-delimiter brackets (kets, bras, floors) possible.

## Use `\left ... \right`, not the fixed `\lvert/\rvert` pairs

`\lvert`/`\rvert` (and `\lfloor/\rfloor`, etc.) are **dedicated fixed pairs**:
`\lvert` is hard-wired to seek a matching `\rvert` and will *not* mate with a
different closer. So `\lvert\psi\rangle` fails - `\lvert` wants `\rvert`,
`\rangle` wants `\langle`, and neither finds its partner.

Any bracket whose two sides differ in shape **must** use the generic
`\left ... \right` mechanism, where `\left`/`\right` open/close with whatever
glyph follows and only the count has to balance.

## Never use package-dependent macros

Office has no package system. Anything that a normal LaTeX document would
pull in from `amsmath`, `braket`, or `physics` simply does not exist in the
build-up engine, and the equation fails.

Never emit these:

```latex
\ket{\psi}
\bra{\psi}
\braket{\phi|\psi}
\lvert\psi\rangle
\langle\psi\rvert
```

Write every bracket out longhand with `\left` and `\right` instead.

## Dirac (bra-ket) notation

| Notation | Office-compatible LaTeX |
| --- | --- |
| Ket | `\left\|\psi\right\rangle` |
| Bra | `\left\langle\psi\right\|` |
| Inner product | `\left\langle\phi\middle\|\psi\right\rangle` |
| Matrix element | `\left\langle\phi\middle\|\hat{A}\middle\|\psi\right\rangle` |
| Ket in a fraction | `\frac{\left\|\psi\right\rangle}{\sqrt{2}}` |

Never write a ket with `\lvert` - always `\left|`.

Use `\middle|` for a bar that sits *inside* a bracket pair, as in an inner
product or a matrix element. Splitting the same expression into two separate
pairs, `\left\langle\phi\right|\hat{A}\left|\psi\right\rangle`, also builds
correctly, but `\middle|` keeps it as one group so every glyph grows to the
same height.

Keep the delimiters explicit inside fractions, where a lone bar does the most
damage:

```latex
\frac{\left\langle\psi\middle|\hat{H}\middle|\psi\right\rangle}
{\left\langle\psi\middle|\psi\right\rangle}
```

## Composite states, outer products, and operators

Write a composite ket as one bracket pair:

```latex
\left|00\right\rangle
```

Keep both pairs when the product structure is what matters:

```latex
\left|0\right\rangle\left|1\right\rangle
```

Write outer products out in full, and wrap the whole outer product in
parentheses when it acts on a ket. Add the parentheses even where they are
not mathematically required - they make the operator-action structure
unambiguous to a reader:

```latex
(\left|0\right\rangle\left\langle1\right|)\left|0\right\rangle
```

Preserve that grouping when expanding the operation:

```latex
(\left|0\right\rangle\left\langle1\right|)\left|0\right\rangle
=
\left|0\right\rangle
\left(\left\langle1\middle|0\right\rangle\right)
=
0
```

Parenthesize a compound operator whenever adjacency could be misread:

```latex
(\hat{A}+\hat{B})\left|\psi\right\rangle
```

A single named operator needs no parentheses:

```latex
\hat{U}\left|\psi\right\rangle
```

## Tensor products

Use `\otimes` when the tensor product should be explicit:

```latex
\left|\psi\right\rangle\otimes\left|\phi\right\rangle
```

Do not silently collapse an explicit tensor product into juxtaposition
unless a shorter form was requested.

## Other Office gotchas

- Absolute value: `\left|x\right|` (stretchy) or `|x|` (fixed size, fine for
  short contents).
- Unsupported LaTeX keywords in Office: `\eqarray`, `\Middle`, `\ldiv`,
  `\dsmash`. Capital `\Middle` is unsupported; lowercase `\middle` is the one
  to use. In the rare case it misbehaves, the fallback is all fixed-size
  brackets with a plain separator, `\langle\phi|\psi\rangle`, which keeps the
  delimiter count balanced.
- Recommended reference: Microsoft's "Linear format equations using UnicodeMath
  and LaTeX in Word" support page.

## Output conventions

When asked for "PowerPoint LaTeX", "Microsoft LaTeX", or "Equation Editor
LaTeX":

1. Put the copyable source in a fenced `latex` code block, raw and
   unrendered, so it can be pasted straight into the equation field.
2. Use explicit `\left ... \right` delimiters and explicit parentheses.
3. Use no package-dependent commands.
4. Do not convert the expression to UnicodeMath unless UnicodeMath was
   specifically requested.
5. Where practical, also show the equation rendered normally so the result
   can be checked by eye.
