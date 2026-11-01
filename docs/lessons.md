# Lessons

What was hard, surprising or worth remembering from the first analyses (2026-10-31: building this repo,
[linear-issue-map](../analyses/linear-issue-map/README.md) and
[linear-universe](../analyses/linear-universe/README.md)). Read this before starting a new analysis; add
to it when something costs you time.

## Harness and tooling

### Getting data

- **Bulk Linear reads go through the API, not the Linear MCP connector.** `list_issues` at 250 per page
  returns ~120k characters, which overflows the tool-result limit; each page lands in a file that then has to
  be parsed. Five pages of that is slow and not reproducible. `analytics fetch linear` (GraphQL with
  `LINEAR_KEY`) pulls all 1,143 issues in 12 requests. Use the MCP connector for small lookups and for writing
  (creating issues, comments).
- **Snapshots are dated** (`data/raw/linear/issues/<date>/`), so a refetch never silently changes an earlier
  result. Analyses read `latest` unless the config pins a date.

### Shell and git in the cloud container

- **`cmd | tail` hides failures.** A failing pytest run was committed because the pipe's exit code was
  `tail`'s. Use `set -o pipefail` (and expect exit 141 from `head` closing a pipe early — that one is
  harmless).
- **`.gitignore` patterns without a leading slash match at any depth.** `data/` also ignored `docs/data/`,
  so the data dictionary silently didn't commit. Anchor root-only rules: `/data/`.
- **uv prints `UV_NATIVE_TLS is deprecated` on every call** in this container. `unset UV_NATIVE_TLS;
  export UV_SYSTEM_CERTS=1` silences it without breaking TLS through the proxy.
- **The GitHub repo was created with a one-line README commit**, so the first push needed a rebase onto
  `origin/main` (keep our README). Check `git ls-remote` before the first push to a new remote.
- **GitHub web pages for repos outside the session return 403** through the container's proxy. That isn't
  a dead link: `git ls-remote https://github.com/<owner>/<repo>` and `raw.githubusercontent.com` still work
  for checking that a repo or file exists.
- **Small container gaps:** `/usr/bin/time` isn't installed (time runs with `date`), `pytest -q` here prints
  no summary line, and pandas 3 reads text columns as `StringDtype` rather than categorical — check
  sort/group behaviour when a pipeline depends on it.
- **Pre-commit hooks are configured but must be installed per clone** (`uv run pre-commit install`). Until
  then, run `ruff` and `pytest` by hand before every commit.

### Dependencies

- **Prototype with `uv run --with <pkg>`** (networkx, manim) instead of adding to `pyproject.toml`; add a
  dependency only after asking, as an optional extra (`graph`, `video`).
- **Manim needs system Pango headers.** `manimpango` has no wheel for this platform and fails with
  `pangocairo >= 1.30.0 is required`; `apt-get install -y libpango1.0-dev` fixes it. The container is
  ephemeral, so this has to be repeated (or added to the environment's setup script). There's no LaTeX
  either: use `Text`, not `MathTex`.

### Multi-agent workflows

- **What it cost.** Designing the universe: 8 agents (3 measure, 3 competing prototypes, 2 judges),
  ~1.7M subagent tokens, ~77 min. Building it: 4 agents (build, visual critic, data verifier, fixer),
  ~1.0M tokens, ~84 min. Porting it into `src/` + an analysis: 3 agents (port, review, fix), ~0.6M
  tokens, ~39 min. Worth it for the open design question (the judges caught real problems: unstable
  galaxies, a misplaced project, catch-all planets); expensive for mechanical steps.
- **Independent review earns its keep.** Reviewers found arrowheads rendering as blobs, labels sitting
  over the wrong planet, a "lint-clean" claim that wasn't, an ablation claim that only held at one level,
  and a held-out check measured against the wrong baseline.
- **Subagents can't write findings or report `.md` files.** Ask for that content in the structured output
  and write the Markdown yourself.
- **Large workflow results get truncated in the notification.** Load the output file as JSON
  (`json.load(...)["result"]`) and print the parts you need.
- **Port with an equivalence bar.** Moving scratch code into the repo was checked by issue id (0 galaxy and
  planet mismatches) and pixel-diffed charts. Compare string columns null-aware: `None`/`NaN`/`NA` render
  differently and look like mismatches. Don't compare outputs while a run is still rewriting them.
- **An analysis can't plain-`import` its own sibling modules**: the CLI loads `run.py` by file path. Use
  `analytics.cli.sibling(__file__, "checks")` (moved there when `linear-expedition` became the second
  analysis to need it — the rule of two in practice).
- **Give agents the facts you already have** (counts, known quirks) so they build on them instead of
  re-deriving; give each a separate `scratch/<folder>/` so parallel agents don't collide.
- **Parallelism is capped by the container's CPUs** (min(16, CPUs − 2)): on this 4-CPU container only two
  agents ran at once, so the third scene group of the video waited for the first two. Plan wall-clock time
  with that in mind, or give fewer, larger units of work.
- **Design workflows to be stoppable.** The video workflow was time-boxed and stopped mid-assembly, before
  its review and fix stages. It was still recoverable in minutes because each scene agent had written a
  self-contained module with "paste from here / until here" markers and no name collisions, and had
  rendered its own drafts. Ask stage outputs to be usable on their own, not only by the next stage.

### Video (ManimCE)

- **Precompute everything in `run.py`, choreograph in `scenes.py`.** Exporting every layout already mapped
  to Manim frame coordinates (`video_data.json`) let three scene agents work in parallel from one data
  contract, and keeps every on-screen number traceable to the analysis outputs.
- **Render at `-ql` while iterating and look at extracted frames** (`ffmpeg -ss <t> -i video.mp4 -frames:v 1`)
  — mid-transition frames are where overlaps hide. A 15 s scene with 1,143 dots renders in ~35 s at 720p30.
- **A video is a lot of choreography code** (~2,100 lines for ten scenes); none of it was reusable yet, so it
  stays in the analysis's one sibling module.

### Charts

- **Always open the PNG.** The first k-means cluster labels piled up in the middle of the t-SNE map; the
  universe needed a label-collision repair loop and a fixed physical scale before it read cleanly.
- **Committed PNGs must stay under 1,000 KB** (pre-commit `check-added-large-files`). The universe chart was
  1.1 MB at 150 dpi; palette quantization (`viz.style.optimize_png`) brings it well under without visible loss.
- **Scatter colors stop at three** (the validated palette's all-pairs limit); fold the rest into "Other" or
  name every group in text, as the universe chart does.
- **Check text against the canvas in code, not only by eye.** `viz.style.texts_outside` reports any text off
  the canvas; longer, more accurate footnotes ran off the edge on first try and the check caught it.
- **Test labels against real geometry.** Bounding boxes of diagonal lines (the 3D range brackets) are useless
  obstacles: they pushed every label away. Test against the segment (`viz.pack.segment_in_box`).
- **On a dense lane chart a label is about one lane tall**: grow the canvas rather than shrink the lanes to
  make room for more text, or labels start landing on the neighbouring lane.
- **Placement costs need graded margins**: a steep penalty for near-misses pushed a callout 215 px from its
  tent. Charge near-misses lightly and true overlaps heavily.
- **`matplotlib.colors.LightSource.hillshade` rescales to 0–1 over the whole image**, so flat ground isn't a
  fixed shade and multiplying RGB greys the plain. Compute the shading yourself and move along the colour
  ramp instead, which also keeps the terrain to one hue.
- **Smoothing at an axis edge needs room**: reflect-mode smoothing folded the first week's mass back and cut
  a peak flat at the start. Empty lead-in weeks (nothing existed yet) are truthful and fix it.

## Methods

- **Count each link once, and check claims against both readings of a link.** A sub-issue whose title
  names its parent was counted as a child and as a mention. "No issue is named twice" held among cairns
  but not among all issues (three open issues are). The accuracy reviewer caught both.
- **Tertile cut-offs over a handful of groups land on ties.** With 7 ranges, Stack's own values became both
  cut-offs, so its "unmarked busy" flag is a tie. Report ties in the data (`on_*_cut` columns), not only in
  prose.
- **Don't seed agents with unverified "facts".** The brief for linear-expedition listed FT2 as stalled; it's
  a completed milestone. The reviewer caught it, but the error came from the orchestrator.

- **t-SNE spreads identical points into a ring.** The 135 near-identical debrief titles can't stack, so
  they form a circle. It's an artifact, not a shape in the data.
- **Cluster in the space you plot.** K-means in 50-dimensional LSA space produced clusters scattered across
  the 2D map; HDBSCAN on the 2D layout gave groups that look like groups.
- **Titles with no shared word can't be placed** by TF-IDF (10 issues, e.g. "Tattoo", hash-only titles).
  Give them NaN, not a fake position, or they collapse to one stray dot.
- **Louvain is seed-sensitive; use a consensus.** Single-seed galaxies moved between runs (ARI as low as
  0.64). Two fixes: a 13-seed consensus, and setting the bridging automation items aside by rule before
  building the graph. Seed ARI rose to 0.99.
- **Check stability three ways:** seeds, 90% subsamples, and one-field ablation. Seeds alone overstated it
  (0.99 vs 0.78-0.96 under subsampling).
- **Held-out checks beat circular ones.** Modularity and "parent pairs kept together" use edges the graph
  was built from. Creation sessions (issues made within 5 minutes) and id-mentions in titles were kept out
  and used as tests — and even those need a same-project baseline, since project is an edge.
- **Name groups from the owner's own words first** (milestone, project, ritual label), then distinctive
  terms, and say "(mixed)" when a group is loose. Two-thirds of human issues (591 of 887) landed in planets
  named by milestone, ritual label or project; 8 planets (270 issues) still needed a short override map.
  Raw term lists like "ai, read, gibson" mean little.

## The data

- **ABC is mostly templates.** 135 `Debrief – <date>` issues (one per day since 2026-06-16, all closed)
  plus a retired *Daily debrief* template (last 2026-06-14). They dominate any text analysis; set them
  aside or filter by team.
- **Delegated items are a pipeline into XYZ.** 103 ABC delegate/capture items, nearly all closed; by text
  they're mostly about Personal OS (49). Many name their target in the title (`… → XYZ-910`): 45 title
  id-mentions in total, a ready-made ground truth for routing.
- **Labels are four different things.** Repo labels (`davidgibsonp/*`) bridge projects (`basecamp` spans
  10 projects); work-type labels (Improvement/Feature/Documentation) are unrelated to topic; origin labels
  (Braindump, Delegate, Pocket, audio-summary) describe how an issue arrived (Braindump titles are less
  alike than random pairs); ritual labels (Retro, Day, Salley-Digest) form tight sub-topics. Split label
  families before using labels for anything.
- **Project is the real structure.** Galaxies are project families (ablation: dropping project changes
  them almost entirely). Big projects are grab-bags in wording: my-system spans 21 of 25 text clusters.
- **Milestones and sub-issue trees carve sub-topics.** 94% of milestone pairs and 97% of parent–child pairs
  stay in one planet. *Life & game* holds together mainly through one tree (XYZ-712, 19 children).
- **Personal OS is 42% of human work**, and the human areas barely link to each other (7 cross-galaxy links).
- **Open work concentrates** in camp (the current frontier), Ideas/reads/job leads, Reading list, and a few
  stalled planets (v1 Compounding harness, open-specification, SOFA, FT2).
- **Two issues have no links at all** (MAIN-21, MAIN-63), and ~10 titles carry no usable text.
