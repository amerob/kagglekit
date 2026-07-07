# Contributing

Contributions that make the skill win more competitions are welcome.

## What to contribute

- **New metric rows** in `references/metric-playbook.md`: metric, best proxy loss, optimal decision rule, legal post-processing.
- **New failure modes**: version traps, silent leakage patterns, API gotchas. These go in the relevant domain file with a one-line reproduction note.
- **Domain deepening**: audio, tabular deep learning, graph ML, protein or molecule competitions, and other areas where the current references are thin.
- **Post-competition evidence**: measured deltas from real competitions that confirm or refute a recommendation in the skill. Evidence beats opinion.

## Ground rules

1. Keep `SKILL.md` under 200 lines. Depth belongs in `references/`.
2. Every technique entry should state when it works, when it fails, and its rough expected impact.
3. Imperative voice, dense prose, no filler.
4. No em-dashes anywhere in any file.
5. One pull request per topic.

## Testing a change

Load the modified skill into a fresh agent session, attach a competition the agent has not seen, and verify it reaches a correct Competition Brief, CV scheme, and decision rule without hand-holding. Note the competition and outcome in the PR description.
