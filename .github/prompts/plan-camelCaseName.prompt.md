## Plan: British English Docstring Revision

TL;DR - correct docstrings and documentation text in `network_diffusion` and `docs/source`into idiomatic British English, while preserving argument and class name tokens.

Steps
1. Inventory all target files: recursively list all `.py` under `network_diffusion` and `.rst`/`.py` (if any) under `docs/source`. Use grep search for triple-quoted strings as initial candidates.
2. Define an American->British spelling dictionary for common terms (e.g., "color"->"colour", "randomize"->"randomise", "optimizer"->"optimiser", etc.). Include likely terms in existing docstrings.
3. Implement a script to process files:
   - For Python files: parse AST to locate module/class/function docstrings. Replace text only inside docstring values. Preserve class/argument names by excluding text inside backticks, code blocks, and names matching identifier patterns (e.g., `CamelCase`, snake_case), and skip transformation within parameter names sections.
   - For `.rst` files: process plain text line-by-line while preserving inline code role segments like ``:class:`Foo`` or ````code```` and not changing identifier tokens.
   - Ensure the script supports dry-run with report of replaced terms and locations.
4. Run the script in dry-run mode, inspect diffs, and adjust dictionary or filtering logic to avoid false positives and unintentional identifier changes.
5. Run the script in apply mode to modify files.
6. Verify with automated checks:
   - `pytest` for regression
   - `python -m compileall network_diffusion docs/source` or equivalent verify syntax after modifications
   - manual spot checks a representative sample of revised docstrings.

Relevant files
- `network_diffusion/**` (all Python modules and packages, especially `network_diffusion/mln`, `network_diffusion/seeding`, `network_diffusion/tpn`)
- `docs/source/**/*.rst` and any `docs/source/**/*.py` if present
- `network_diffusion/logger.py`, `simulator.py`, `metrics.py`, etc. as high-priority docstring files

Verification
1. Dry-run script output should show changed docstring lines; shifted metrics should indicate exact matches.
2. Functional tests via `pytest` pass after transformation.
3. Document build (Sphinx) for docs/source if project supports it.

Decisions
- Exclude argument/class names by skipping formatted code segments and identifiers.
- Keep docstring meaning and structure unchanged; only spelling/phrase variations to British usage.

Further Considerations
1. If a strict policy is desired on British idioms (beyond spelling), define style rules up front, e.g., "ensure `analyse` and `optimise` are used".
2. For ambiguous terms, follow consistency with existing British-style words appearing already in repository.
