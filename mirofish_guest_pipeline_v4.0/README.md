# MiroFish hotel guest profiles: v4 experience-state generator

This package updates your **existing** Python guest-profile script, adding an `experience_state_model.py` that constructs guest-specific circumstances from your observed ticket taxonomy. It **does not** require rebuilding `ml_data.mirofish_guest_profiles_v1`.

## Run

1. Run `00_build_guest_profiles_redshift.sql` to (re)build `ml_data.mirofish_guest_profiles_v1`, then export `01_export_guest_profiles.sql` to `guest_profiles.csv`. The export's row count should equal the table's row count.
2. In the package folder:

```bash
pip install pandas
python 02_generate_synthetic_guests.py \
  --csv guest_profiles.csv \
  --ticket-csv ticket_mix.csv \
  --n 1000 --seed 42 \
  --scenario-fraction 0.20 \
  --scenario-severity moderate \
  --scenario-resolution unresolved \
  --impact-strength moderate \
  --out-dir ./out
```

The 20% of stays assigned controlled experiences is a **deliberate experiment setting**, **not** the observed 18% *per eligible guest message* ticket detection rate.

`simulation_assumptions.json` also gets a `response_calibration` block: the provisional reply-rate target used by engagement-message experiments (see below). It defaults to 3,128 replies / 17,593 stays receiving a status check = 17.78%, anchored to the `rating` variant. Override it with `--calibration-responses`, `--calibration-checks` and `--calibration-baseline`.

## What's new

- For a selected ticket taxonomy event, constructs `experience_state` with kind (repair/problem vs routine request), assigned severity, resolution, and pre-message context.
- Constructs `agent_context = persona_text + stay context`: **this is the field to give the agent** when integrating with MiroFish.
- Examples: *AC repair*: a modeled problem that noticeably interferes with comfort; *additional towel request*: a routine need, **not automatically a negative hotel experience**.
- Includes a configurable `relative_satisfaction_pressure` **only for modeled problems**. These are arbitrary sensitivity-test coefficients, **not empirical impacts**, not normalized satisfaction values, and are **not included in `agent_context`**.
- Keeps all historical status-check scores out of the agent input. They go only to `out/historical_outcomes_for_calibration.csv`.
- For guests without injected experiences, no specific prior stay experience is claimed. This **does not** mean the stay was issue-free.

## Outputs

| File | Purpose |
|---|---|
| `out/synthetic_guests.jsonl` | Agent-ready *input records*; includes `persona_text`, `experience_state`, and `agent_context` |
| `out/synthetic_guests_summary.csv` | Quick review of scenario assignments |
| `out/historical_outcomes_for_calibration.csv` | Scores/scales (kept entirely separate from agent JSONL) |
| `out/status_check_score_distribution.csv` | Observed score mix **among scored responses** only |
| `out/simulation_assumptions.json` | Controls and provenance |

## Scope and caveats

- The MiroFish adapter loads guests as agents (see above); engagement-message experiments are not implemented yet.
- No real guest/stay issue probabilities or satisfaction-penalty estimates are claimed.
- Ticket frequencies are conditional on **detected recommendations in three hotels** and need hotel-offering filters or property-specific weights for broad deployment.
- Results should be compared across sensitivity settings (e.g., `--impact-strength weak` vs `strong`) rather than interpreting a single assumed number as an effect size.
- `01_export_guest_profiles.sql` assumes the table was rebuilt with the current `00_build_guest_profiles_redshift.sql` (it reads `status_check_score`, `status_check_scale` and `scored_message_count`).
- Score and scale are taken together from the first scored message in each tour. `scored_message_count > 1` marks tours with multiple scores; review them before using the score distributions as calibration targets.
- Profiles are sampled without replacement when the CSV has at least `--n` rows, so no source row is duplicated into identical agents.

## Loading into MiroFish

From `backend/`:

```bash
uv run python scripts/build_guest_simulation.py --guests ../mirofish_guest_pipeline_v4.0/out/synthetic_guests.jsonl
```

This writes a ready-to-run simulation folder under `backend/uploads/simulations/` and prints the commands to verify and run it. Only `agent_context` reaches the agents; the full records are kept in `population.json` for analysis.

## Population API (saved populations and persona enrichment)

The backend (default `http://localhost:5001`) can store a population once, optionally enrich its personas with the LLM, and create any number of simulations from it with identical agents.

```bash
API=http://localhost:5001/api/population

# 1. Upload (records are validated; the population can't be edited afterwards)
curl -F guests=@out/synthetic_guests.jsonl \
     -F assumptions=@out/simulation_assumptions.json \
     -F name="Q4 guests" $API/upload
# -> {"population_id": "pp_...", "guest_count": 100, "default_seed": 42, ...}

# 2. Optional: enrich personas (background task; costs one LLM call per guest)
curl -X POST -H 'Content-Type: application/json' \
     -d '{"parallel": 5}' $API/pp_xxx/enrich          # -> {"task_id": "..."}
curl $API/task/<task_id>                               # progress / result
curl $API/pp_xxx/audit                                 # trait mix, fallbacks, flags
curl "$API/pp_xxx/guests?limit=5&cohort=controlled_scenario"

# 3. Create a simulation (template or enriched personas); this freezes the population
curl -X POST -H 'Content-Type: application/json' \
     -d '{"persona_source": "enriched", "hours": 24}' $API/pp_xxx/simulations
# -> {"simulation_id": "pop_...", "random_seed": 42, ...}

# 4. Run it with the existing simulation API (platform "twitter" runs the
#    parallel script with --twitter-only, which writes twitter/actions.jsonl)
curl -X POST -H 'Content-Type: application/json' \
     -d '{"simulation_id": "pop_...", "platform": "twitter", "enable_graph_memory_update": false}' \
     http://localhost:5001/api/simulation/start
```

Add `-H 'Accept-Language: en'` to curl calls for English error messages; the backend defaults to Chinese.

How enrichment is bounded:

- Each agent's text is `persona_text` (observed, verbatim) + `ADDITIONAL TRAITS (synthetic):` paragraph + `CURRENT STAY CONTEXT:` experience (verbatim). The LLM writes only the middle paragraph.
- Python assigns the traits (formality, tone, detail, priority) with a seeded generator; the LLM only words them. An observed `observed_writing_style` replaces the sampled detail value. Override the default uniform weights with `"trait_weights": {"tone": {"warm": 1, "impatient": 1}}`.
- Responsiveness, helpfulness and willingness to give feedback are never assigned or described: replying is the outcome being measured.
- The LLM never sees the stay experience, `company_id`, satisfaction-pressure values or historical scores.
- Paragraphs with cooperative language, reply/feedback wording, ratings, compensation, demographics, a wrong night count or another sub-brand are retried once, then fall back to the template persona and are flagged in `/audit`.
- Enrichment re-runs only guests that aren't enriched yet (pass `"force": true` to redo all). Once a simulation exists, the population is frozen and enrichment returns 409; upload again to try different settings.

What is and isn't controlled across simulations of one population:

- Identical: guest personas, stay experiences (severity and resolution included), simulation time settings you pass, and which agents are activated each round (seeded by `random_seed`, defaulting to the population's seed).
- Not controlled: the LLM's sampling during the run (Claude Haiku 5.5 on Bedrock rejects `temperature`, and Bedrock has no seed) and OASIS's internal feed ordering. Compare variants with repeated runs rather than single runs.

## Engagement-message experiments (direct engine)

The direct engine sends each message variant privately to every guest in a population, as a single text message from the hotel. There is no feed and no other guests, so the message is the only thing that differs between variants.

Whether a guest replies is decided in two stages, because asking the LLM to "respond or ignore" made it reply almost every time:

1. **Propensity (LLM):** the guest only states how likely they would be to reply, `reply_likelihood` from 0 to 100, without writing anything. It is clipped to `p_raw` in [0.01, 0.99].
2. **Calibration (Python):** `p_calibrated = sigmoid(logit(p_raw) + b)`. One intercept `b` is fitted so the baseline variant's (`rating`) mean `p_calibrated` equals the target reply rate, and that same `b` is applied to every variant. Variants are never calibrated separately. This keeps each guest's ordering and the odds ratios between variants, not absolute differences.
3. **Draw (Python):** `u = Random(f"{seed}:{guest_id}:{replicate}")`, identical for every variant. The guest responds when `u < p_calibrated`, so a variant with a higher probability for that guest can only turn an ignore into a reply, never the reverse.
4. **Writing (LLM):** a separate call writes the reply, only for guests drawn to respond.

Every result row keeps `propensity_raw`, `p_raw`, `p_calibrated`, `calibration_intercept` and `random_draw`; stage-1 answers are also saved in `propensities.jsonl`. Errors record `error_stage` (`propensity` or `writing`).

```bash
EXP=http://localhost:5001/api/experiments

# 1. Start an experiment (background task). This freezes the population and
#    snapshots the exact agent texts and experiences before any LLM call.
curl -X POST -H 'Content-Type: application/json' -H 'Accept-Language: en' -d '{
  "population_id": "pp_xxx",
  "persona_source": "enriched",
  "engine": "direct",
  "replicates": 1,
  "analyze": true,
  "reasoning_effort": "high",
  "response_calibration": {"target_reply_rate": 0.1778, "baseline_variant_id": "rating"},
  "variants": [
    {"variant_id": "checkin", "text": "Hi from {brand}! How is everything with your room so far?"},
    {"variant_id": "rating",  "text": "How would you rate your stay so far, 1-5?", "score_scale": 5},
    {"variant_id": "open",    "text": "Is there anything we can do to make your stay better?"}
  ]
}' $EXP
# -> {"experiment_id": "ex_...", "task_id": "...", "expected_rows": 30, "snapshot_sha256": "...",
#     "calibration_status": "provisional_target", "warnings": []}

curl $EXP/task/<task_id>                          # progress by stage: propensity, writing, coding
curl $EXP/ex_xxx                                  # spec, hashes, calibration and fitted intercept
curl $EXP/ex_xxx/summary                          # per-variant results and comparisons
curl "$EXP/ex_xxx/results?variant=rating&decision=respond&limit=20"
curl "$EXP/ex_xxx/results?flag=unsupported_incident"

# Optional: code replies afterwards (or again with "force": true)
curl -X POST -H 'Content-Type: application/json' -d '{}' $EXP/ex_xxx/analyze
```

- **Variants:** 1-5 per experiment. `{brand}` is replaced with the guest's sub-brand, or "the hotel". Set `score_scale` (5 or 10) only for variants that ask for a number; a score is recorded only when the reply contains exactly one unambiguous value on that scale.
- **Rows:** guests x variants x replicates. Errors (two invalid LLM responses in a row at either stage) are counted separately and excluded from rates.
- **Calls:** one propensity call per row, plus one writing call per responder. At a 17.78% target, 30 rows cost about 35 guest calls plus coding.
- **Blind coding** (`analyze`): a separate LLM call sees only the reply text and a fixed category list (every assigned need category in the population, plus `other`). It codes the feedback type, stated issues, whether staff could act on the reply (`actionable`), and any specific incident the reply reports. A Python step then compares those with each guest's assigned need: issue discovery (out of all controlled-scenario guests, including those who ignored the message), issues raised that weren't assigned, and `unsupported_incident`, an incident the experience state doesn't establish. An unsupported incident isn't proof of fabrication; the persona text may imply it.
- **Metrics per variant:** mean `propensity_raw` and `p_calibrated`, simulated reply rate, **actionable feedback yield per message sent** (actionable replies divided by all non-error messages, including ignores), unsupported incidents (count and rate per reply), scores and issue discovery.
- **Cohorts:** guests are reported in three cohorts derived from the assigned need's `need_kind`: `unspecified` (no assigned experience), `service_request` (the guest wants or needs something) and `repair_problem` (a property-controlled failure: a repair or complaint ticket type). Each variant's `by_cohort` gives raw and calibrated propensity, reply rate and actionable yield, and each comparison's `within_cohort` repeats the paired differences on that cohort's guests. `guests_by_cohort` shows how many guests each cohort has; compare messages within a cohort rather than relying only on the overall mix.
- **Intervals:** every 95% interval is a seeded guest-level bootstrap: a guest's replicates are averaged first, then guests are resampled. Each metric reports `instances` (rows) and `unique_guests`; the guest count is the real sample size.
- **Comparisons:** paired on the same guests, for reply rate, `p_calibrated`, actionable yield and mean raw log-odds.
- **Reproducibility:** each experiment keeps `snapshot.jsonl` (exact agent text, experience, assigned need and a hash per guest) plus `snapshot_sha256`, the population's `personas_sha256`, prompt versions, reasoning effort, the resolved calibration settings, the fitted intercept and the git commit. Runs read only from the snapshot. Experiments on the same frozen population have the same snapshot hash.

Where the calibration target comes from, in order:

1. `response_calibration` in the request, field by field (for example `{"target_reply_rate": 0.20}` keeps the population's baseline).
2. `response_calibration` in the population's `simulation_assumptions.json` (written by the generator).
3. Neither: no calibration. The response includes a warning, the summary reports `not_calibrated`, and `p_calibrated` equals `p_raw`.

The baseline variant must be one of the experiment's variants, otherwise the request is rejected. `provisional` defaults to `true`, and a target set in the request is always provisional. The summary's `calibration.status` is `provisional_target`, `calibrated` (only when `provisional` is `false`) or `not_calibrated`.

**The 17.78% target is provisional.** Before treating it as official, verify that the 3,128 numerator counts guest replies to delivered status checks, not only replies with a valid score. If it only counts scored replies, the true reply rate is higher.

How to read the results:

- Results are for **picking promising A/B candidates**, not validated treatment effects. The calibration anchors overall reply frequency on the baseline only. It is not a validated population response model and doesn't establish how much a different message changes response.
- The shared draw makes variants comparable guest by guest, so reply differences come only from how the message changed the stated propensity. Whether those propensity differences match real guests is untested.
- Satisfaction-score calibration is deferred to V2 (`score_calibration: "deferred_v2"`).
- Start with a smoke test (10 guests, 3 variants, 1 replicate). Check that every variant has the same snapshot hash, the baseline's mean `p_calibrated` is close to the target, `ignore` rows have no text, and the draws match across variants. Then rerun with `"replicates": 3` before comparing variants.
- The `oasis_social` engine (guests and staff interacting in OASIS, with groups isolated per variant) is planned as a separate experiment mode and currently returns an error.
