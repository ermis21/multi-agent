### skill_install
Install a skill from a GitHub folder URL (e.g. an `anthropics/skills` folder). Downloads `SKILL.md` plus any supporting files into `config/skills/{name}/`. Writes are atomic — if `SKILL.md` fails frontmatter validation nothing lands on disk. Pass `overwrite: true` to replace an existing skill of the same name. Returns `{name, dir, files}`.
<|tool_call|>call: skill_install, {"url": "https://github.com/anthropics/skills/tree/main/skills/<folder>"}<|tool_call|>
