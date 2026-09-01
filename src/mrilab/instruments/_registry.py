"""Instrument registry — metadata only, importable from the SERVICE venv.

No torch here (enforced by tests/unit/test_import_hygiene.py): the service uses
this table to plan jobs (lane, dtype, pool dependencies) without importing any
instrument. The hidden cross-module file contracts of the research repo are made
explicit in `produces`/`consumes` (pool filenames are CONTENT KEYS — exact
upstream patterns, resolved relative to the executor's runtime cwd under
experiments/regimes/).

dtype notes: "fp32" modules load AutoModelForCausalLM(dtype=torch.float32)
explicitly — the 2e-4-nat noise floor all grids are calibrated against is an
fp32 number. "bf16" modules go through ModelInspector, whose dtype="auto"
resolves to bfloat16. kd_arms is mixed (bf16 teacher, fp32 student).
"""

from __future__ import annotations

INSTRUMENTS: dict[str, dict] = {
    "vitals": {
        "module": "mrilab.instruments.vitals",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "vitals_{label}.json",
        "produces": [],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    "sensitivity_ladder": {
        "module": "mrilab.instruments.sensitivity_ladder",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "row_{label}.json",
        "produces": [],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    "icl_readiness": {
        "module": "mrilab.instruments.icl_readiness",
        "lane": "cpu",  # upstream default device is cpu; gpu-able
        "dtype": "fp32",
        "output": "row_{revision}.json",
        "produces": [],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    "icl_ladder": {
        "module": "mrilab.instruments.icl_ladder",
        "lane": "gpu",
        "dtype": "fp32",  # optional --quant 4bit path uses bitsandbytes
        "output": "row_{label}.json",
        "produces": [],
        "consumes": [],
        "network": ["allenai/sciq", "pminervini/HaluEval", "truthfulqa/truthful_qa"],
    },
    "honesty_battery": {
        "module": "mrilab.instruments.honesty_battery",
        "lane": "gpu",
        "dtype": "bf16",  # ModelInspector path
        "output": "honesty_battery.json",
        "produces": [],
        "consumes": [],
        "network": ["mandarjoshi/trivia_qa", "allenai/sciq", "google-research-datasets/nq_open"],
    },
    "pref_battery": {
        "module": "mrilab.instruments.pref_battery",
        "lane": "gpu",
        "dtype": "bf16",
        "output": "pref_battery.json",
        "produces": [],
        "consumes": [],
        "network": ["(prefs.SOURCES: 17 HF preference datasets)"],
    },
    "m1_signals": {
        "module": "mrilab.instruments.m1_signals",
        "lane": "gpu",
        "dtype": "bf16",
        "output": "m1_signals.json",
        "produces": [],
        "consumes": [],
        "network": ["allenai/c4", "nyu-mll/blimp"],
    },
    "induction_ladder": {
        "module": "mrilab.instruments.induction_ladder",
        "lane": "gpu",
        "dtype": "bf16",
        "output": "induction_{label}.json",
        "produces": [],
        "consumes": [],
        "network": [],
    },
    "erosion_arms": {
        "module": "mrilab.instruments.erosion_arms",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "erosion.json",
        # cache key: _pool_{target}_{n_train + n_target_eval}_{seq_len}.pt
        "produces": ["_pool_{target}_{n}_512.pt"],
        "consumes": [],
        "network": ["allenai/c4", "bigcode/the-stack-smol", "pile-of-law/pile-of-law",
                    "hoskinson-center/proof-pile", "ashraq/financial-news-articles"],
    },
    "repetition_arms": {
        "module": "mrilab.instruments.repetition_arms",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "erosion.json",
        "produces": ["_pool_c4rep_16032_512.pt"],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    "kd_arms": {
        "module": "mrilab.instruments.kd_arms",
        "lane": "gpu",
        "dtype": "mixed",  # bf16 teacher, fp32 student
        "output": "kd.json",
        "produces": ["_pool_kd_c4_{n}_512.pt"],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    "l8_arms": {
        "module": "mrilab.instruments.l8_arms",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "l8.json",
        "produces": [],
        # written ONLY by kd_arms.pool_for — running l8 first raises upstream
        "consumes": ["_pool_kd_c4_4128_512.pt"],
        "network": [],
    },
    "s_cross": {
        "module": "mrilab.instruments.s_cross",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "row_{label}.json",
        "produces": [],
        # written ONLY by erosion_arms --target code --n-train 8192
        "consumes": ["_pool_code_8224_512.pt"],
        "network": ["allenai/c4"],
    },
    "dp_profile": {
        "module": "mrilab.instruments.dp_profile",
        "lane": "cpu",  # upstream default device is cpu
        "dtype": "fp32",
        "output": "row_{corpus}_{model}.json",
        "produces": ["_dp_pool_{corpus}_1024_512.pt"],
        "consumes": [],
        "network": ["(all domain corpora)"],
    },
    "z_ter": {
        "module": "mrilab.instruments.z_ter",
        "lane": "cpu",  # RTN quantization damage; never touches CUDA
        "dtype": "fp32",
        "output": "row_{label}.json",
        "produces": [],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    "u1_run": {
        "module": "mrilab.instruments.u1_run",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "u1.json",
        "produces": ["_pool_u1_40960_512.pt", "_pool_u1_rule_512.pt"],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    "match_capability": {
        "module": "mrilab.instruments.match_capability",
        "lane": "gpu",
        "dtype": "fp32",
        "output": "match.json",
        "produces": [],
        "consumes": [],
        "network": ["allenai/c4"],
    },
    # libraries (no CLI entrypoint; imported by the drivers above):
    #   prefs, payload_ladder, icl semantics live inside icl_ladder runs.
}

LIBRARIES = ("prefs", "payload_ladder")
