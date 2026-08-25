# skills/

Project skills for llama.cpp work live here. Skills that are **not** llama.cpp-specific have
been moved to `~/.claude/skills/`, where they are available in every project.

## Moved out (2026-08-23)

- **`macos-reversing`** -> `~/.claude/skills/macos-reversing/`. Driving Apple's private
  ObjC frameworks from Python via ctypes to reach undocumented formats and APIs. Nothing in
  it is llama.cpp-specific; it was written here only because that is where the GPU counter
  and replay work happened. Docs in `perf/` that say "read `skills/macos-reversing`" mean
  that path.

  **The moved copy is the complete one.** Three branches each carried a partial SKILL.md
  because agent working trees were destroyed twice by branch switching and rebuilt from
  context. The global copy is the union: 21 sections, including the three
  (`A/B the input before believing a theory about it`, `Handing a file to a sandboxed
  helper`, `Plugin bundles load themselves once you ask for one`) that existed only on
  `dy-headless-replay`. Do not restore any single branch's version over it.

## Still here, because they are llama.cpp-specific

- `add-new-model` - adding a model architecture to llama.cpp.
- `code-review` - llama.cpp conventions and reviewer pitfalls.

## Symlinked out (2026-08-25), canonical copy stays here

- `metal-gpu-profile` and `metal-kernel-prescreen` are installed as symlinks:
  `~/.claude/skills/<name> -> ~/play/llama.cpp-prod/skills/<name>`. The canonical copy is
  this one, on `prod` - iterate here and the installed skill updates with it. Each skill
  carries its scripts in its own `references/` directory; the old `perf/<name>` paths
  still work as symlinks into `references/`, so perf docs that cite them stay correct.
  The capture and compile steps still assume a fork checkout as cwd; the SKILL.mds say so.
