# QIS101

Claude Code does not read `AGENTS.md` natively, so this file imports it.
The line below pulls in the full style guide.

@AGENTS.md

Keep all style and formatting rules in `AGENTS.md` so that every AI tool used
with this repository follows the same guide. Add rules here only if they are
specific to Claude Code and meaningless to other tools.

## Chat Responses

Responses are read in the Claude Code panel inside VS Code, which has no
MathJax or KaTeX renderer. LaTeX written there shows up as literal dollar
signs and backslashes, so it is unreadable.

- In chat replies, write math as plain text: `2^n`, `P(q7 = 1)`, `<=`,
  `|psi>`, `sqrt(2)`.
- Do not use `$...$` or `$$...$$` in chat replies.
- This rule applies **only** to chat. Markdown cells in `.ipynb` files render
  LaTeX correctly in the Jupyter view, so keep proper `$...$` math there.

## Git Commits

Commit messages must carry no trace of AI assistance. Write the message as
the repository owner would have written it.

- Do not add a `Co-Authored-By: Claude ...` trailer, or any similar
  "co-authored by Claude" line.
- Do not add a generated-by footer such as
  "Generated with Claude Code" or a robot emoji tag.
- Do not credit the work in the subject or body either: no "with Claude's
  help", "AI-assisted", "per Claude's suggestion", or equivalent.

The ban is on **attribution**, not on the words themselves. When a change is
genuinely *about* Claude or AI tooling, name it plainly, because that is what
the commit is:

- Good: `Add anthropic.claude-code to the extension install scripts`
- Good: `Document the chat math convention in CLAUDE.md`
- Bad: `Update install scripts (generated with Claude Code)`
