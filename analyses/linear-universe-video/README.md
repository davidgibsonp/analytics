# linear-universe-video

A ~4.5-minute animated walkthrough of [linear-universe](../linear-universe/README.md): how 1,143 Linear issues
go from a cloud of titles to 8 galaxies, and a tour of what's in each.

## Question

Can the method behind the universe chart be understood without reading the code, and can the result be
explored galaxy by galaxy instead of all at once?

## Answer

The video makes one argument in ten scenes, each proving one step (no narration; captions carry the story):

| # | Scene | ~s | What it shows |
|---|---|---|---|
| 1 | Cloud | 15 | 1,143 issues at random positions, by team |
| 2 | Words → map | 35 | titles → TF-IDF terms; issues fly to the t-SNE map; 25 loose clusters, the debrief ring |
| 3 | Wording isn't enough | 20 | my-system issues scatter across 21 of 25 text clusters |
| 4 | The graph | 35 | a small subset with each edge type added in turn: text neighbours, project, milestone, parent, sibling, repo label |
| 5 | Hub + communities | 35 | automation set aside by rule; single seeds vs the 13-seed consensus; 8 galaxies |
| 6 | Planets & names | 25 | one galaxy splits into planets; the naming cascade |
| 7 | The universe | 30 | every issue flies from the word map to its place in the universe; hub arrows |
| 8 | Tour | 50 | one stop per galaxy, each with its stats and one story |
| 9 | What each field adds | 20 | the one-field ablation as bars (galaxy and planet ARI) |
| 10 | Close | 10 | caveats: galaxies are project families, positions carry no meaning |

Every number on screen comes from `linear-universe`'s `metrics.json` and `assignments.parquet` or
`linear-issue-map`'s outputs, exported by `run.py`; nothing is typed into the scenes by hand.

## Evidence

- `figures/linear-universe-preview.mp4`: the whole video (4:35) as a 769 KB preview, 640 px wide at 10 fps.
  Captions are readable; the dots smear. Watch the full render for detail.
- `figures/poster-{2,5,7,9}-*.png`: stills of the word map, the consensus, the universe and the ablation.
- Full render: `outputs/linear-universe.mp4` (1280x720, 30 fps, ~19 MB, ~5 min to render on 4 CPUs),
  not committed.

## Method

- **`run.py`** exports everything the scenes need into `outputs/video_data.json`, with all layouts
  (random, t-SNE, universe, a spring layout of the issue graph) already mapped to Manim frame coordinates,
  then renders each scene class with Manim's Python API and joins them with ffmpeg.
- **`scenes.py`** holds the shared style and helpers at the top and the ten scene classes below.
  It is long (~2,100 lines) because each scene is choreography; there is no reusable logic worth moving to
  `src/` yet.
- **`config.toml`** has render quality, per-scene target durations, colors, font sizes, the scene-4 subset
  and the tour order.
- Style: dark surface, the palette's dark-mode steps for galaxies in the same order as the universe chart,
  captions in neutral ink, `Text` only (no LaTeX).

## Status and known gaps

This cut was built by three scene agents in parallel and merged; the planned independent pacing review and
fact-check pass did **not** run (the work was time-boxed). Known rough spots:

- Cross-fades between tour stops (scene 8) briefly overlap the outgoing planet label with the scene title.
- Pacing hasn't been checked against reading speed by a fresh viewer.
- A few scene tunables still live as module constants at the top of each scene group in `scenes.py`
  (`S1_SIDES`, `S7_FLY_ORDER`, `TOUR_*`, …) rather than in `config.toml`.

## Rerun

```bash
sudo apt-get install -y libpango1.0-dev pkg-config   # once per machine: manimpango builds from source
uv sync --extra graph --extra video
uv run --extra graph analytics run linear-universe         # the data the video reads
uv run analytics run linear-issue-map
uv run --extra graph --extra video analytics run linear-universe-video --set render.quality=medium_quality
# preview for the repo (must stay under the 1,000 KB pre-commit limit; 480p was 1.5 MB even at crf 38):
ffmpeg -y -i analyses/linear-universe-video/outputs/linear-universe.mp4 -vf scale=640:-2,fps=10 -c:v libx264 \
  -crf 40 -preset veryslow -tune animation -pix_fmt yuv420p -an -movflags +faststart \
  analyses/linear-universe-video/figures/linear-universe-preview.mp4
```

`render.quality` defaults to `high_quality` (1080p); `medium_quality` (720p30) is faster. One scene can be
rendered alone with `--set 'render.scenes=["S8Tour"]'`.
