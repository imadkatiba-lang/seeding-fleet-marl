"""seeding-fleet-marl (single-file version)

Multi-agent reinforcement learning for the dispatch of a cloud-seeding fleet over Morocco and the adjacent
Atlantic: glaciogenic seeding over the Atlas and hygroscopic seeding offshore, under uncertainty about the
seeding effect. Six aircraft (one per airport) are trained with MAPPO, IPPO or a centralised PPO controller in
an environment that replays hourly ERA5 fields, and compared with a greedy and a random rule by
leave-one-season-out validation.

Code of the paper: Katiba, I., Belmajdoub, H., Minaoui, K. "Coordinating a seeding-aircraft fleet with
multi-agent reinforcement learning under effect uncertainty: glaciogenic seeding over the Atlas and
hygroscopic seeding over the Moroccan Atlantic." Submitted to Atmospheric Research.
Repository: https://github.com/imadkatiba-lang/seeding-fleet-marl (MIT licence).

Contents
    1. Configuration and paths
    2. Environment (ERA5 replay, seedability masks, bounded effect, randomised reserve)
    3. Networks and full-season evaluation of a policy
    4. Training on one leave-one-season-out fold
    5. Paired multi-seed evaluation and summary statistics
    6. Complete experimental protocol, data check, figures
    7. Command line / Jupyter entry point

Folders (next to this file, or in the current folder in Jupyter; override with environment variables):
    data/seasons/   season_<year>_<year+1>.nc   (SEEDMARL_DATA)
    results/        outputs                     (SEEDMARL_RESULTS)

Command line:
    python seeding_fleet_marl.py check --year 2011
    python seeding_fleet_marl.py train --algo mappo --test-year 2009 --seed 0
    python seeding_fleet_marl.py experiments        # full protocol (60 runs, resumable)
    python seeding_fleet_marl.py evaluate           # paired evaluation of all runs + summary
    python seeding_fleet_marl.py summary            # summary of results/evaluation_multiseed.csv
    python seeding_fleet_marl.py figures            # figures of the paper (needs cartopy)

Jupyter: run the whole file as one cell after setting TASK (and its options) in section 7.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import gaussian_kde


# =====================================================================================================
#  1. Configuration and paths
# =====================================================================================================
DEFAULT_CONFIG = {
    "description": "Multi-agent dispatch of a cloud-seeding fleet over Morocco and the adjacent Atlantic (glaciogenic seeding over the Atlas, hygroscopic seeding offshore).",
    "domain": {
        "north": 36.0,
        "west": -16.75,
        "south": 21.5,
        "east": -2.0,
        "resolution_deg": 0.25,
        "grid": [
            59,
            60
        ]
    },
    "seasons": {
        "2009": "wet",
        "2011": "dry",
        "2012": "normal",
        "2015": "dry",
        "2025": "wet"
    },
    "months": "December-March",
    "data": {
        "source": "ERA5 hourly reanalysis (Hersbach et al., 2020), single and pressure levels, accessed through the Destination Earth Data Hub",
        "file_pattern": "season_{year}_{next_year}.nc",
        "dynamic_variables": [
            "u10",
            "v10",
            "d2m",
            "t2m",
            "msl",
            "tp",
            "cp",
            "lsp",
            "tcc",
            "lcc",
            "mcc",
            "hcc",
            "tclw",
            "tciw",
            "tcwv",
            "blh",
            "cape",
            "kx",
            "t700",
            "u700",
            "v700",
            "w700",
            "q700",
            "z500",
            "ivt_e",
            "ivt_n",
            "ivt",
            "sp"
        ],
        "static_variables": [
            "orog",
            "lsm"
        ],
        "excluded_variables": {
            "cin": "undefined outside convective conditions (97.6 % missing values)"
        },
        "ivt_method": "Trapezoidal integral of q*V over 1000, 925, 850, 700, 600, 500, 400 and 300 hPa; levels below the surface (p > sp) are excluded."
    },
    "seedability": {
        "min_terrain_height_m": 1000.0,
        "glaciogenic_masks": {
            "G1": {
                "t700_max_C": 0.0,
                "tclw_min": 0.05,
                "w700_max": 0.0
            },
            "G2": {
                "t700_max_C": -2.0,
                "tclw_min": 0.05,
                "w700_max": 0.0
            },
            "G3": {
                "t700_max_C": -5.0,
                "tclw_min": 0.05,
                "w700_max": 0.0
            },
            "G4": {
                "t700_max_C": -5.0,
                "tclw_min": 0.1,
                "w700_max": -0.1
            },
            "G5": {
                "t700_max_C": -8.0,
                "tclw_min": 0.1,
                "w700_max": -0.2
            }
        },
        "hygroscopic_masks": {
            "H1": {
                "tclw_min": 0.05,
                "max_distance_km": 300,
                "ivt_min": 0
            },
            "H2": {
                "tclw_min": 0.1,
                "max_distance_km": 200,
                "ivt_min": 100
            },
            "H3": {
                "tclw_min": 0.1,
                "max_distance_km": 200,
                "ivt_min": 250
            },
            "H4": {
                "tclw_min": 0.2,
                "max_distance_km": 150,
                "ivt_min": 250
            }
        },
        "selected": {
            "glaciogenic": "G3",
            "hygroscopic": "H3"
        },
        "sensitivity_variants": {
            "glaciogenic": [
                "G2",
                "G4"
            ],
            "hygroscopic": [
                "H2",
                "H4"
            ]
        },
        "rationale": {
            "glaciogenic": "T700 <= -5 C: activation threshold of silver iodide; T700 is a conservative proxy for cloud temperature.",
            "hygroscopic": "IVT >= 250 kg m-1 s-1: usual atmospheric-river threshold; onshore transport required (ivt_e > 0)."
        }
    },
    "seeding_effect": {
        "form": "gain = alpha * seedability_score * natural_precipitation, applied in seedable cells and advected downwind with (u700, v700)",
        "alpha_unit": "fraction of the natural precipitation of the seeded clouds during the event (not of the seasonal total)",
        "max_glaciogenic_mm_per_h": 0.3,
        "training_scenario": "pessimistic",
        "alpha": {
            "glaciogenic": {
                "none": 0.0,
                "pessimistic": 0.03,
                "central": 0.05,
                "optimistic": 0.14
            },
            "hygroscopic": {
                "none": 0.0,
                "pessimistic": 0.0,
                "central": 0.03,
                "optimistic": 0.1
            }
        },
        "sources": {
            "glaciogenic_none": "WWMPP: no statistically significant effect (p = 0.28), Rasmussen et al. (2018)",
            "glaciogenic_pessimistic": "WWMPP ensemble modelling, lower quartile (3 %), Rasmussen et al. (2018)",
            "glaciogenic_central": "WWMPP ensemble modelling, mean effect of 5 % per seedable event, Rasmussen et al. (2018)",
            "glaciogenic_optimistic": "Snowy Mountains (SPERP), +14 % on suitable events, Manton et al. (2017)",
            "glaciogenic_cap": "SNOWIE: 0.05-0.3 mm of snow per event of about one hour, Friedrich et al. (2020)",
            "hygroscopic_pessimistic": "WMO (2025) and Silverman (2003): effect not convincingly demonstrated",
            "hygroscopic_central_optimistic": "Working hypotheses set below the glaciogenic values and well below the exploratory radar estimates of Mather et al. (1997)",
            "hygroscopic_delay": "Mather et al. (1997): cell response 20-60 min after seeding"
        }
    },
    "environment": {
        "bases": {
            "Casablanca": [
                33.37,
                -7.59
            ],
            "Rabat": [
                34.05,
                -6.75
            ],
            "Tangier": [
                35.73,
                -5.92
            ],
            "Marrakech": [
                31.61,
                -8.04
            ],
            "Agadir": [
                30.33,
                -9.41
            ],
            "Dakhla": [
                23.72,
                -15.93
            ]
        },
        "aircraft_per_base": {
            "Casablanca": 1,
            "Rabat": 1,
            "Tangier": 1,
            "Marrakech": 1,
            "Agadir": 1,
            "Dakhla": 1
        },
        "speed_kmh": 350.0,
        "endurance_h": 4.5,
        "refuel_h": 1,
        "k_candidates": 8,
        "patch": 5,
        "episode_h": 72,
        "cape_max": 1500.0,
        "score_refs": {
            "glaciogenic_tclw": 0.2,
            "glaciogenic_w700": 0.5,
            "hygroscopic_tclw": 0.3,
            "hygroscopic_ivt": 500.0
        },
        "delays_h": {
            "glaciogenic": [
                [
                    1,
                    1.0
                ]
            ],
            "hygroscopic": [
                [
                    1,
                    0.3333333333333333
                ],
                [
                    2,
                    0.3333333333333333
                ],
                [
                    3,
                    0.3333333333333333
                ]
            ]
        },
        "reserve": {
            "r_min": 0.1,
            "r_max": 0.5
        },
        "costs": {
            "flight_Mm3_per_h": 0.001,
            "seeding_Mm3": 0.0003
        }
    },
    "training": {
        "algorithms": [
            "mappo",
            "ippo",
            "ppo"
        ],
        "iterations": 300,
        "episodes_per_iteration": 16,
        "evaluate_every": 50,
        "lr_actor": 0.0003,
        "lr_critic": 0.001,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip": 0.2,
        "epochs": 5,
        "minibatches": 4,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "reward_scale": 100.0,
        "hidden_units": 256
    },
    "evaluation": {
        "scenarios": [
            "pessimistic",
            "central",
            "optimistic"
        ],
        "train_seeds": [
            0,
            1,
            2
        ],
        "eval_seeds": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10
        ],
        "bootstrap_resamples": 20000,
        "bootstrap_seed": 2026
    }
}

ROOT = Path(os.environ.get("SEEDMARL_ROOT", Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()))
DATA_DIR = Path(os.environ.get("SEEDMARL_DATA", ROOT / "data" / "seasons"))
RESULTS_DIR = Path(os.environ.get("SEEDMARL_RESULTS", ROOT / "results"))
RUNS_DIR = RESULTS_DIR / "runs"

#: optional override: a JSON file with the same structure as DEFAULT_CONFIG
CONFIG_FILE = os.environ.get("SEEDMARL_CONFIG")
CFG = json.loads(Path(CONFIG_FILE).read_text(encoding="utf-8")) if CONFIG_FILE else copy.deepcopy(DEFAULT_CONFIG)

#: test seasons of the leave-one-season-out folds, keyed by first year, with their regime
SEASONS = {int(year): regime for year, regime in CFG["seasons"].items()}
#: fold order used to run the experiments (the near-normal season first)
FOLDS = [2012, 2009, 2011, 2015, 2025]
#: season order used for statistics and figures (wet to dry); it fixes the bootstrap resampling of the paper
SEASON_ORDER = [2009, 2025, 2012, 2015, 2011]


def season_file(year):
    """Path of the unified ERA5 file of the season starting in December of `year`."""
    path = DATA_DIR / f"season_{year}_{year + 1}.nc"
    if not path.exists():                       # name used by earlier versions of the data files
        legacy = DATA_DIR / f"saison_{year}_{year + 1}.nc"
        if legacy.exists():
            return legacy
    return path


def season_label(year):
    """'2009/10' for the season starting in December 2009."""
    return f"{year}/{str(year + 1)[2:]}"


def run_name(policy, test_year, seed):
    """Folder name of one training run, e.g. 'mappo_test2009_seed0' or 'ppo600_test2011_seed2'."""
    return f"{policy}_test{test_year}_seed{seed}"


# =====================================================================================================
#  2. Environment
# =====================================================================================================
TECHNIQUES = ("glaciogenic", "hygroscopic")
ENV_PARAMS = CFG["environment"]


def default_params():
    """Independent copy of the environment parameters of config.json."""
    return copy.deepcopy(ENV_PARAMS)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Grid:
    """Fixed geometry of the domain: candidate areas, distances and cell areas."""

    def __init__(self, season_path):
        with xr.open_dataset(season_path) as d:
            self.lat = d.latitude.values.astype(float)
            self.lon = d.longitude.values.astype(float)
            self.lsm = d.lsm.values > 0.5
            self.orog = d.orog.values.astype(float)
        self.ny, self.nx = self.lsm.shape
        LA, LO = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.LA, self.LO = LA, LO
        land = (LA[self.lsm], LO[self.lsm])
        self.coast_distance = np.array([haversine_km(a, b, *land).min()
                                        for a, b in zip(LA.ravel(), LO.ravel())]).reshape(LA.shape)
        s = CFG["seedability"]
        self.glaciogenic_area = self.lsm & (self.orog >= s["min_terrain_height_m"])
        h = s["hygroscopic_masks"][s["selected"]["hygroscopic"]]
        self.hygroscopic_area = (~self.lsm) & (self.coast_distance <= h["max_distance_km"])
        dlat_km = abs(self.lat[1] - self.lat[0]) * 111.2
        dlon_km = abs(self.lon[1] - self.lon[0]) * 111.2 * np.cos(np.deg2rad(LA))
        self.cell_area_m2 = dlat_km * dlon_km * 1e6
        self.y_km = (self.lat[0] - LA) * 111.2                  # planar positions for fast distances
        self.x_km = (LO - self.lon[0]) * 111.2 * np.cos(np.deg2rad(LA))
        self.dy_km, self.dx_km = dlat_km, float(np.mean(dlon_km))

    def cell(self, lat, lon):
        return int(np.abs(self.lat - lat).argmin()), int(np.abs(self.lon - lon).argmin())

    def km(self, a, b):
        return float(haversine_km(self.lat[a[0]], self.lon[a[1]], self.lat[b[0]], self.lon[b[1]]))


def load_season(path, grid, params=None):
    """Load one season and precompute the seedability masks and monotone scores in [0, 1]."""
    p = params or ENV_PARAMS
    names = ["tp", "tclw", "t700", "w700", "ivt", "ivt_e", "u700", "v700", "cape"]
    with xr.open_dataset(path) as d:
        D = {v: d[v].values.astype(np.float32) for v in names}
        time = d.time.values
    s = CFG["seedability"]
    mg_cfg = s["glaciogenic_masks"][s["selected"]["glaciogenic"]]
    mh_cfg = s["hygroscopic_masks"][s["selected"]["hygroscopic"]]
    t700_C = D["t700"] - 273.15
    safe = D["cape"] <= p["cape_max"]
    mask_g = (grid.glaciogenic_area[None] & (t700_C <= mg_cfg["t700_max_C"]) & (D["tclw"] >= mg_cfg["tclw_min"])
              & (D["w700"] <= mg_cfg["w700_max"]) & safe)
    mask_h = (grid.hygroscopic_area[None] & (D["tclw"] >= mh_cfg["tclw_min"]) & (D["ivt"] >= mh_cfg["ivt_min"])
              & (D["ivt_e"] > 0) & safe)
    r = p["score_refs"]
    score_g = np.clip(D["tclw"] / r["glaciogenic_tclw"], 0, 1) * np.clip(-D["w700"] / r["glaciogenic_w700"], 0, 1)
    score_h = np.clip(D["tclw"] / r["hygroscopic_tclw"], 0, 1) * np.clip(D["ivt"] / r["hygroscopic_ivt"], 0, 1)
    return {"time": time, "tp_mm": D["tp"] * 1000.0, "u700": D["u700"], "v700": D["v700"], "ivt": D["ivt"],
            "mask_g": mask_g, "mask_h": mask_h,
            "score_g": np.where(mask_g, score_g, 0).astype(np.float32),
            "score_h": np.where(mask_h, score_h, 0).astype(np.float32)}


class SeedingEnv:
    """Parallel multi-agent environment: one agent per seeding aircraft."""

    def __init__(self, seasons, grid, scenario="pessimistic", params=None, seed=0, mode="train"):
        assert mode in ("train", "evaluate"), mode
        self.seasons, self.g, self.p, self.mode = seasons, grid, params or ENV_PARAMS, mode
        eff = CFG["seeding_effect"]
        self.alpha = {t: eff["alpha"][t][scenario] for t in TECHNIQUES}
        self.max_glaciogenic_mm = eff["max_glaciogenic_mm_per_h"]
        self.rng = np.random.default_rng(seed)
        self.agents, self.home = [], {}
        for base, n in self.p["aircraft_per_base"].items():
            for i in range(n):
                a = f"{base}_{i + 1}"
                self.agents.append(a)
                self.home[a] = grid.cell(*self.p["bases"][base])
        self.K = self.p["k_candidates"]
        self.n_actions = self.K + 2
        self.range_km = self.p["speed_kmh"]
        self.obs_dim = 7 + 4 * self.K + 3 + 2 * self.p["patch"] ** 2
        self.state_dim = self.obs_dim * len(self.agents)

    # ------------------------------------------------------------------ episode
    def reset(self, season=None, start=None):
        self.s = self.seasons[season] if season is not None else self.seasons[self.rng.integers(len(self.seasons))]
        T = len(self.s["time"])
        L = self.p["episode_h"] if self.mode == "train" else T - 4
        self.t0 = int(start) if start is not None else int(self.rng.integers(0, max(1, T - L - 4)))
        self.t, self.end = self.t0, min(self.t0 + L, T - 4)
        self.pos = {a: self.home[a] for a in self.agents}
        self.endurance = {a: self.p["endurance_h"] for a in self.agents}
        self.on_ground = {a: True for a in self.agents}
        self.refuel_left = {a: 0 for a in self.agents}
        self.log = []                          # one row per seeding decision (for IPW estimation)
        self.stats = {"gain_Mm3": 0.0, "flight_hours": 0.0, "seedings": 0, "reserved": 0}
        self._candidates()
        return self._observations(), self._infos()

    def _gain(self, cell, technique, score, t):
        """Rain gained over land (million m3) by one seeding, displaced downwind at 700 hPa."""
        i, j = cell
        u, v = float(self.s["u700"][t, i, j]), float(self.s["v700"][t, i, j])
        total, land_mm = 0.0, 0.0
        for h, weight in self.p["delays_h"][technique]:
            th = t + h
            if th >= len(self.s["time"]):
                break
            ii = int(round(i - v * 3.6 * h / self.g.dy_km))           # v > 0 northwards: latitude index decreases
            jj = int(round(j + u * 3.6 * h / self.g.dx_km))
            if not (0 <= ii < self.g.ny and 0 <= jj < self.g.nx):
                continue
            mm = self.alpha[technique] * score * weight * float(self.s["tp_mm"][th, ii, jj])
            if technique == "glaciogenic":
                mm = min(mm, self.max_glaciogenic_mm)
            if self.g.lsm[ii, jj]:
                land_mm += mm
                total += mm / 1000.0 * float(self.g.cell_area_m2[ii, jj]) / 1e6
        return total, land_mm

    def _candidates(self):
        """For each aircraft: the K best seedable cells within reach this hour."""
        t = self.t
        sg, sh = self.s["score_g"][t], self.s["score_h"][t]
        score = np.maximum(sg, sh)
        tech = np.where(sg >= sh, 0, 1)
        ii, jj = np.nonzero(score > 0)
        self.opportunities = (ii, jj, score[ii, jj], tech[ii, jj])
        self.cands = {}
        for a in self.agents:
            if self.refuel_left[a] > 0 or len(ii) == 0:
                self.cands[a] = []
                continue
            pi, pj = self.pos[a]
            hi, hj = self.home[a]
            d = np.hypot(self.g.x_km[ii, jj] - self.g.x_km[pi, pj], self.g.y_km[ii, jj] - self.g.y_km[pi, pj])
            d_home = np.hypot(self.g.x_km[ii, jj] - self.g.x_km[hi, hj], self.g.y_km[ii, jj] - self.g.y_km[hi, hj])
            # reachable this hour AND return possible with the remaining endurance (one-hour margin)
            ok = (d <= self.range_km) & (d_home / self.range_km + 1.0 <= self.endurance[a] - 1.0 + 1e-6)
            idx = np.nonzero(ok)[0]
            idx = idx[np.argsort(-self.opportunities[2][idx])][: self.K]
            self.cands[a] = [(int(ii[k]), int(jj[k]), float(self.opportunities[2][k]), int(self.opportunities[3][k]),
                              float(d[k])) for k in idx]

    def action_mask(self, a):
        m = np.zeros(self.n_actions, dtype=bool)
        if self.refuel_left[a] > 0:
            m[self.K + 1] = True
            return m
        m[: len(self.cands[a])] = True
        d_home = self.g.km(self.pos[a], self.home[a])
        must_return = (not self.on_ground[a]) and (d_home / self.range_km + 1.0 > self.endurance[a] - 1.0)
        if must_return:
            m[:] = False
        elif len(self.opportunities[0]) > 0:
            m[self.K] = True                    # reposition
        m[self.K + 1] = True                    # return / stay on the ground: always allowed
        return m

    def _infos(self):
        return {a: {"action_mask": self.action_mask(a)} for a in self.agents}

    def _patch(self, field, c):
        r = self.p["patch"] // 2
        padded = np.pad(field, r)
        return padded[c[0]: c[0] + 2 * r + 1, c[1]: c[1] + 2 * r + 1].ravel()

    def _observations(self):
        t = self.t
        hour = (np.datetime64(self.s["time"][t], "h").astype(int) % 24) / 24.0
        n_g = float(self.s["mask_g"][t].sum()) / max(1, self.g.glaciogenic_area.sum())
        n_h = float(self.s["mask_h"][t].sum()) / max(1, self.g.hygroscopic_area.sum())
        ivt_mean = float(self.s["ivt"][t].mean()) / 500.0
        obs = {}
        for a in self.agents:
            c, b = self.pos[a], self.home[a]
            own = [self.endurance[a] / self.p["endurance_h"], float(self.on_ground[a]),
                   math.sin(2 * math.pi * hour), math.cos(2 * math.pi * hour),
                   (self.g.lat[c[0]] - self.g.lat.min()) / np.ptp(self.g.lat),
                   (self.g.lon[c[1]] - self.g.lon.min()) / np.ptp(self.g.lon),
                   self.g.km(c, b) / self.range_km]
            cand = np.zeros((self.K, 4), dtype=np.float32)
            for k, (i, j, sc, te, d) in enumerate(self.cands[a]):
                cand[k] = [sc, d / self.range_km, te, 1.0]
            local = np.concatenate([self._patch(self.s["score_g"][t], c), self._patch(self.s["score_h"][t], c)])
            obs[a] = np.concatenate([np.array(own, np.float32), cand.ravel(),
                                     np.array([n_g, n_h, ivt_mean], np.float32), local.astype(np.float32)])
        return obs

    def global_state(self, obs):
        return np.concatenate([obs[a] for a in self.agents])

    # ------------------------------------------------------------------ time step
    def step(self, actions):
        p, t = self.p, self.t
        r, c = p["reserve"], p["costs"]
        seeded_cells, gain, flights, seedings = set(), 0.0, 0.0, 0
        for a in self.agents:
            act = int(actions[a])
            if not self.action_mask(a)[act]:
                act = self.K + 1                                      # invalid action -> return
            if self.refuel_left[a] > 0:                               # refuelling on the ground
                self.refuel_left[a] -= 1
                if self.refuel_left[a] == 0:
                    self.endurance[a] = p["endurance_h"]
                continue
            if act < self.K:                                          # seed a candidate
                i, j, sc, te, d = self.cands[a][act]
                self.pos[a] = (i, j)
                self.on_ground[a] = False
                if (i, j) not in seeded_cells:                        # a cell is treated once per hour
                    seeded_cells.add((i, j))
                    technique = TECHNIQUES[te]
                    p_reserve = r["r_max"] - (r["r_max"] - r["r_min"]) * sc
                    reserved = bool(self.rng.random() < p_reserve)
                    g_cf, mm = self._gain((i, j), technique, sc, t)    # counterfactual gain if seeded
                    self.log.append({"t": str(self.s["time"][t])[:13], "aircraft": a, "i": i, "j": j,
                                     "lat": float(self.g.lat[i]), "lon": float(self.g.lon[j]),
                                     "technique": technique, "score": sc, "p_reserve": p_reserve,
                                     "reserved": reserved, "gain_if_seeded_Mm3": g_cf,
                                     "gain_if_seeded_mm_land": mm})
                    if reserved:
                        self.stats["reserved"] += 1
                    else:
                        gain += g_cf
                        seedings += 1
            elif act == self.K:                                       # reposition towards the best opportunity
                ii, jj, sc, _ = self.opportunities
                k = int(np.argmax(sc))
                self._move(a, (int(ii[k]), int(jj[k])))
                self.on_ground[a] = False
            else:                                                     # return or stay on the ground
                if not self.on_ground[a]:
                    self._move(a, self.home[a])
                    if self.pos[a] == self.home[a]:
                        self.on_ground[a] = True
                        self.refuel_left[a] = p["refuel_h"]
            if not self.on_ground[a]:
                self.endurance[a] -= 1.0
                flights += 1.0
        reward = gain - c["flight_Mm3_per_h"] * flights - c["seeding_Mm3"] * seedings
        self.stats["gain_Mm3"] += gain
        self.stats["flight_hours"] += flights
        self.stats["seedings"] += seedings
        self.t += 1
        done = self.t >= self.end
        self._candidates()
        obs = self._observations()
        rewards = {a: reward for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncated = {a: done for a in self.agents}
        infos = self._infos()
        for a in self.agents:
            infos[a]["gain_Mm3"] = gain
        return obs, rewards, terminated, truncated, infos

    def _move(self, a, target):
        c = self.pos[a]
        d = self.g.km(c, target)
        if d <= self.range_km:
            self.pos[a] = target
        else:
            f = self.range_km / d
            self.pos[a] = (int(round(c[0] + f * (target[0] - c[0]))), int(round(c[1] + f * (target[1] - c[1]))))


# ----------------------------- baseline policies -----------------------------
def random_policy(env, obs, infos, rng):
    """Uniformly random valid action for every aircraft."""
    return {a: int(rng.choice(np.nonzero(infos[a]["action_mask"])[0])) for a in env.agents}


def greedy_policy(env, obs, infos, rng):
    """Each aircraft seeds its best candidate (no coordination), otherwise returns or stays on the ground."""
    return {a: (0 if infos[a]["action_mask"][0] else env.K + 1) for a in env.agents}


# =====================================================================================================
#  3. Networks and full-season evaluation of a policy
# =====================================================================================================
ALGORITHMS = ("mappo", "ippo", "ppo")
HIDDEN = CFG["training"]["hidden_units"]


def mlp(n_in, n_out, hidden=HIDDEN):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def build_actor(algo, obs_dim, state_dim, n_agents, n_actions):
    return mlp((state_dim if algo == "ppo" else obs_dim) + n_agents, n_actions)


def build_critic(algo, obs_dim, state_dim, n_agents):
    return mlp(obs_dim + n_agents, 1) if algo == "ippo" else mlp(state_dim, 1)


def actor_inputs(env, obs, algo):
    """One input row per aircraft: its local observation, or the global state for the centralised controller."""
    n = len(env.agents)
    if algo == "ppo":
        return np.broadcast_to(env.global_state(obs).astype(np.float32), (n, env.state_dim)).copy()
    return np.stack([obs[a] for a in env.agents]).astype(np.float32)


def load_actor(path, algo, env, device="cpu"):
    """Load the actor of a saved run (also reads checkpoints written by earlier versions of the code)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    weights = ckpt["actor"] if "actor" in ckpt else ckpt["acteur"]
    actor = build_actor(algo, env.obs_dim, env.state_dim, len(env.agents), env.n_actions).to(device)
    actor.load_state_dict(weights)
    return actor.eval()


def play_season(test_seasons, grid, scenario, seed, policy, params, actor=None, algo=None, device="cpu"):
    """Run one policy deterministically over a whole test season.

    policy: "learned" (needs `actor` and `algo`), "greedy" or "random".
    Returns the season statistics (with the net water balance and the glaciogenic share) and the decision log.
    """
    env = SeedingEnv(test_seasons, grid, scenario, params, seed=seed, mode="evaluate")
    rng = np.random.default_rng(seed)
    ids = torch.eye(len(env.agents), device=device)
    obs, infos = env.reset(season=0, start=0)
    while True:
        if policy == "learned":
            x = torch.as_tensor(actor_inputs(env, obs, algo), device=device)
            mask = torch.as_tensor(np.stack([infos[a]["action_mask"] for a in env.agents]), device=device)
            with torch.no_grad():
                act = actor(torch.cat([x, ids], -1)).masked_fill(~mask, -1e9).argmax(-1).cpu().numpy()
            actions = {a: int(v) for a, v in zip(env.agents, act)}
        elif policy == "greedy":
            actions = greedy_policy(env, obs, infos, rng)
        else:
            actions = random_policy(env, obs, infos, rng)
        obs, _, _, truncated, infos = env.step(actions)
        if all(truncated.values()):
            stats = dict(env.stats)
            log = pd.DataFrame(env.log)
            c = params["costs"]
            stats["net_balance_Mm3"] = (stats["gain_Mm3"] - c["flight_Mm3_per_h"] * stats["flight_hours"]
                                        - c["seeding_Mm3"] * stats["seedings"])
            executed = ~log.reserved if len(log) else None
            stats["glaciogenic_share"] = (float((log.loc[executed, "technique"] == "glaciogenic").mean())
                                          if len(log) and executed.any() else 0.0)
            return stats, log


# =====================================================================================================
#  4. Training on one leave-one-season-out fold
# =====================================================================================================
def train(algo="mappo", test_year=2012, seed=0, iterations=None, label=None, train_scenario=None,
          flight_cost_m3_per_h=None, verbose=True, **overrides):
    """Train one policy with the test season held out; returns the run folder."""
    assert algo in ALGORITHMS, algo
    hp = {**CFG["training"], **overrides}
    if iterations is not None:
        hp["iterations"] = iterations
    train_scenario = train_scenario or CFG["seeding_effect"]["training_scenario"]
    label = label or algo
    run = {"algo": algo, "label": label, "test_year": test_year, "seed": seed, "train_scenario": train_scenario,
           "eval_scenarios": CFG["evaluation"]["scenarios"], **hp}
    log = print if verbose else (lambda *a, **k: None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = RUNS_DIR / run_name(label, test_year, seed)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Device: {device} | fold: test {season_label(test_year)} ({SEASONS[test_year]}) | output: {out}")

    params = default_params()
    if flight_cost_m3_per_h is not None:
        params["costs"]["flight_Mm3_per_h"] = flight_cost_m3_per_h / 1e6
    run["costs"] = params["costs"]
    grid = Grid(season_file(test_year))
    t0 = time.time()
    train_seasons = [load_season(season_file(y), grid, params) for y in SEASONS if y != test_year]
    test_seasons = [load_season(season_file(test_year), grid, params)]
    log(f"Data loaded in {time.time() - t0:.0f} s: {len(train_seasons)} training seasons, 1 test season")

    env = SeedingEnv(train_seasons, grid, train_scenario, params, seed=seed, mode="train")
    N, A, OD, SD = len(env.agents), env.n_actions, env.obs_dim, env.state_dim
    actor = build_actor(algo, OD, SD, N, A).to(device)
    critic = build_critic(algo, OD, SD, N).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=hp["lr_actor"])
    opt_critic = torch.optim.Adam(critic.parameters(), lr=hp["lr_critic"])
    ids = torch.eye(N, device=device)

    def masked_logits(x, mask):
        logits = actor(torch.cat([x, ids.expand(x.shape[0] // N, N, N).reshape(-1, N)], -1))
        return logits.masked_fill(~mask, -1e9)

    def to_arrays(obs, infos):
        return (np.stack([obs[a] for a in env.agents]).astype(np.float32),
                np.stack([infos[a]["action_mask"] for a in env.agents]))

    @torch.no_grad()
    def value(o, s):
        if algo == "ippo":                     # one value per aircraft, from its local observation
            return critic(torch.cat([torch.as_tensor(o, device=device), ids], -1)).squeeze(-1).cpu().numpy()
        return critic(torch.as_tensor(s, device=device)).item()

    @torch.no_grad()
    def sample(o, m, s):
        x = np.broadcast_to(s, (N, s.shape[0])).copy() if algo == "ppo" else o
        logits = masked_logits(torch.as_tensor(x, device=device), torch.as_tensor(m, device=device))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy()

    def test(policy, scenario):
        return play_season(test_seasons, grid, scenario, 1, policy, params, actor=actor, algo=algo, device=device)

    # ----------------------------- training loop -----------------------------
    curve, t_start = [], time.time()
    for it in range(1, hp["iterations"] + 1):
        O, S, MSK, ACT, LP, R, V, D, gains = [], [], [], [], [], [], [], [], []
        for _ in range(hp["episodes_per_iteration"]):
            obs, infos = env.reset()
            while True:
                o, m = to_arrays(obs, infos)
                s = env.global_state(obs).astype(np.float32)
                a, lp = sample(o, m, s)
                v = value(o, s)
                obs, r, _, truncated, infos = env.step({ag: int(x) for ag, x in zip(env.agents, a)})
                done = all(truncated.values())
                O.append(o); S.append(s); MSK.append(m); ACT.append(a); LP.append(lp)
                R.append(r[env.agents[0]] * hp["reward_scale"]); V.append(v); D.append(done)
                if done:
                    gains.append(env.stats["gain_Mm3"])
                    break

        # generalised advantage estimation on the shared team reward
        R, V, D = np.array(R, np.float32), np.array(V, np.float32), np.array(D)   # IPPO: V has shape (T, N)
        adv = np.zeros_like(V)
        g = np.zeros_like(V[0]) if algo == "ippo" else 0.0
        for t in reversed(range(len(R))):
            v_next = 0.0 if D[t] else V[t + 1]
            delta = R[t] + hp["gamma"] * v_next - V[t]
            g = delta + hp["gamma"] * hp["gae_lambda"] * (0.0 if D[t] else g)
            adv[t] = g
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = len(R)
        O_t = torch.as_tensor(np.concatenate(O), device=device)                # (T*N, obs_dim)
        M_t = torch.as_tensor(np.concatenate(MSK), device=device)
        A_t = torch.as_tensor(np.concatenate(ACT), device=device)
        LP_t = torch.as_tensor(np.concatenate(LP), device=device)
        S_t = torch.as_tensor(np.stack(S), device=device)
        if algo == "ippo":                       # advantage and return of each aircraft
            ADV_t = torch.as_tensor(adv.reshape(-1), device=device)
            RET_t = torch.as_tensor(ret.reshape(-1), device=device)
        else:
            ADV_t = torch.as_tensor(np.repeat(adv, N), device=device)
            RET_t = torch.as_tensor(ret, device=device)

        idx = np.arange(T)
        for _ in range(hp["epochs"]):
            np.random.shuffle(idx)
            for mb in np.array_split(idx, hp["minibatches"]):
                mb_t = torch.as_tensor(mb, device=device)
                rows = (mb_t[:, None] * N + torch.arange(N, device=device)[None]).reshape(-1)
                X = S_t[mb_t].repeat_interleave(N, 0) if algo == "ppo" else O_t[rows]
                dist = torch.distributions.Categorical(logits=masked_logits(X, M_t[rows]))
                ratio = torch.exp(dist.log_prob(A_t[rows]) - LP_t[rows])
                surr1 = ratio * ADV_t[rows]
                surr2 = torch.clamp(ratio, 1 - hp["clip"], 1 + hp["clip"]) * ADV_t[rows]
                loss_actor = -torch.min(surr1, surr2).mean() - hp["entropy_coef"] * dist.entropy().mean()
                opt_actor.zero_grad(); loss_actor.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), hp["max_grad_norm"]); opt_actor.step()
                if algo == "ippo":
                    v_pred = critic(torch.cat([O_t[rows], ids.expand(len(mb), N, N).reshape(-1, N)], -1)).squeeze(-1)
                    loss_critic = hp["value_coef"] * ((v_pred - RET_t[rows]) ** 2).mean()
                else:
                    loss_critic = hp["value_coef"] * ((critic(S_t[mb_t]).squeeze(-1) - RET_t[mb_t]) ** 2).mean()
                opt_critic.zero_grad(); loss_critic.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), hp["max_grad_norm"]); opt_critic.step()

        row = {"iteration": it,
               "mean_episode_return_Mm3": float(R.sum() / hp["episodes_per_iteration"] / hp["reward_scale"]),
               "mean_episode_gain_Mm3": float(np.mean(gains)), "actor_loss": loss_actor.item(),
               "critic_loss": loss_critic.item(), "minutes": (time.time() - t_start) / 60}
        if it % hp["evaluate_every"] == 0 or it == hp["iterations"]:
            st, _ = test("learned", train_scenario)
            row.update({"test_gain_Mm3": st["gain_Mm3"], "test_net_balance_Mm3": st["net_balance_Mm3"],
                        "test_flight_hours": st["flight_hours"]})
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "config": run}, out / "model.pt")
        curve.append(row)
        if it % 10 == 0 or it == 1 or "test_gain_Mm3" in row:
            msg = (f"it {it:4d}/{hp['iterations']} | return/episode {row['mean_episode_return_Mm3']:+.4f} Mm3 | "
                   f"gain/episode {row['mean_episode_gain_Mm3']:.4f} Mm3 | {row['minutes']:.1f} min")
            if "test_gain_Mm3" in row:
                msg += (f" -> test season: gain {row['test_gain_Mm3']:.2f} Mm3, net {row['test_net_balance_Mm3']:+.2f}, "
                        f"{row['test_flight_hours']:.0f} flight hours")
            log(msg, flush=True)
            pd.DataFrame(curve).to_csv(out / "learning_curve.csv", index=False)

    # ----------------------------- final test-season evaluation -----------------------------
    rows = []
    for sc in run["eval_scenarios"]:
        for policy in ["learned", "greedy", "random"]:
            st, decisions = test(policy, sc)
            if policy == "learned":
                decisions.to_csv(out / f"decisions_{sc}.csv", index=False)
            rows.append({"test_season": season_label(test_year), "regime": SEASONS[test_year], "scenario": sc,
                         "policy": label if policy == "learned" else policy,
                         "gain_Mm3": round(st["gain_Mm3"], 3), "flight_hours": int(st["flight_hours"]),
                         "seedings": st["seedings"], "reserved": st["reserved"],
                         "glaciogenic_share": round(st["glaciogenic_share"], 2),
                         "net_balance_Mm3": round(st["net_balance_Mm3"], 3),
                         "gain_per_flight_hour_m3": round(st["gain_Mm3"] * 1e6 / max(st["flight_hours"], 1)),
                         "train_scenario": train_scenario, "iterations": hp["iterations"], "seed": seed})
    res = pd.DataFrame(rows)
    res.to_csv(out / "evaluation.csv", index=False)
    (out / "params.json").write_text(json.dumps(run, indent=2), encoding="utf-8")   # written last
    log(res[["scenario", "policy", "gain_Mm3", "flight_hours", "net_balance_Mm3"]].to_string(index=False))
    log(f"Done in {(time.time() - t_start) / 60:.0f} min -> {out}")
    return out


# =====================================================================================================
#  5. Paired multi-seed evaluation and summary statistics
# =====================================================================================================
EVAL_FILE = RESULTS_DIR / "evaluation_multiseed.csv"
SUMMARY_FILE = RESULTS_DIR / "summary.csv"
SCENARIOS = CFG["evaluation"]["scenarios"]
EVAL_SEEDS = CFG["evaluation"]["eval_seeds"]
TRAIN_SEEDS = CFG["evaluation"]["train_seeds"]
#: learned policies: label -> algorithm
LEARNED = {"mappo": "mappo", "ippo": "ippo", "ppo": "ppo", "ppo600": "ppo"}
BASELINES = ["greedy", "random"]
POLICIES = list(LEARNED) + BASELINES
PAIRS = [("mappo", "ippo"), ("mappo", "ppo"), ("mappo", "ppo600"), ("ippo", "ppo"), ("ppo600", "ppo"),
         ("mappo", "greedy"), ("mappo", "random"), ("ippo", "greedy"), ("ppo", "greedy")]


def evaluate_all(verbose=True):
    """Evaluate every completed run (and the baselines) on every evaluation seed and scenario."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = default_params()
    done = set()
    if EVAL_FILE.exists():
        d = pd.read_csv(EVAL_FILE)
        done = set(zip(d.fold, d.scenario, d.eval_seed, d.policy, d.train_seed))
    candidates = [(lab, g) for lab in LEARNED for g in TRAIN_SEEDS] + [(b, -1) for b in BASELINES]
    t_start, n_done = time.time(), 0
    for year in FOLDS:
        available = [(lab, g) for lab, g in candidates
                     if g < 0 or (RUNS_DIR / run_name(lab, year, g) / "params.json").exists()]
        todo = [(sc, ge, lab, g) for sc in SCENARIOS for ge in EVAL_SEEDS for lab, g in available
                if (year, sc, ge, lab, g) not in done]
        if not todo:
            continue
        grid = Grid(season_file(year))
        test = [load_season(season_file(year), grid, params)]
        env0 = SeedingEnv(test, grid, "central", params, mode="evaluate")
        actors = {(lab, g): load_actor(RUNS_DIR / run_name(lab, year, g) / "model.pt", LEARNED[lab], env0, device)
                  for lab, g in available if g >= 0}
        if verbose:
            print(f"Season {season_label(year)} ({SEASONS[year]}): {len(todo)} evaluations", flush=True)
        for sc in SCENARIOS:
            for ge in EVAL_SEEDS:
                block = [x for x in todo if x[0] == sc and x[1] == ge]
                rows = []
                for _, _, lab, g in block:
                    learned = lab in LEARNED
                    st, _ = play_season(test, grid, sc, ge, "learned" if learned else lab, params,
                                        actor=actors.get((lab, g)), algo=LEARNED.get(lab), device=device)
                    rows.append({"fold": year, "test_season": season_label(year), "regime": SEASONS[year],
                                 "scenario": sc, "eval_seed": ge, "policy": lab, "train_seed": g,
                                 "gain_Mm3": round(st["gain_Mm3"], 4), "flight_hours": int(st["flight_hours"]),
                                 "seedings": st["seedings"], "reserved": st["reserved"],
                                 "glaciogenic_share": round(st["glaciogenic_share"], 3),
                                 "net_balance_Mm3": round(st["net_balance_Mm3"], 4)})
                if rows:
                    pd.DataFrame(rows).to_csv(EVAL_FILE, mode="a", header=not EVAL_FILE.exists(), index=False)
                    n_done += len(rows)
                    if verbose:
                        print(f"   {sc:<11} seed {ge:2d} | " + " | ".join(
                            f"{r['policy']}{'' if r['train_seed'] < 0 else r['train_seed']} {r['net_balance_Mm3']:+.2f}"
                            for r in rows) + f" | {(time.time() - t_start) / 60:.0f} min", flush=True)
    return n_done


def summarise(verbose=True):
    """Mean net balance, paired differences with 95 % bootstrap intervals over seasons, seed stability."""
    d = pd.read_csv(EVAL_FILE)
    d = d[d.fold.isin(FOLDS) & d.eval_seed.isin(EVAL_SEEDS) & ((d.train_seed < 0) | d.train_seed.isin(TRAIN_SEEDS))]
    policies = [p for p in POLICIES if p in set(d.policy)]
    per_season = d.groupby(["scenario", "policy", "fold"])[
        ["net_balance_Mm3", "gain_Mm3", "flight_hours", "seedings", "glaciogenic_share"]].mean()
    rows = []
    table = per_season["net_balance_Mm3"].groupby(["scenario", "policy"]).mean().unstack()[policies].reindex(SCENARIOS)
    for sc in SCENARIOS:
        for p in policies:
            rows.append({"table": "net_balance", "scenario": sc, "comparison": p, "value": table.loc[sc, p]})
    ev = CFG["evaluation"]
    rng = np.random.default_rng(ev["bootstrap_seed"])
    resample = rng.integers(0, len(SEASON_ORDER), size=(ev["bootstrap_resamples"], len(SEASON_ORDER)))
    nb = per_season["net_balance_Mm3"].unstack("policy")
    for sc in SCENARIOS:
        t = nb.loc[sc].reindex(SEASON_ORDER)
        for a, b in PAIRS:
            if a in t and b in t and t[a].notna().all() and t[b].notna().all():
                x = (t[a] - t[b]).values
                lo, hi = np.percentile(x[resample].mean(1), [2.5, 97.5])
                rows.append({"table": "paired_difference", "scenario": sc, "comparison": f"{a} - {b}",
                             "value": x.mean(), "ci_low": lo, "ci_high": hi, "seasons_won": int((x > 0).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_FILE, index=False)
    if verbose:
        pd.set_option("display.width", 200)
        print("\nMean net water balance (Mm3 per season), averaged over the five held-out seasons")
        print(table.round(2).to_string())
        print("\nFlight hours per season:", per_season["flight_hours"].groupby("policy").mean().reindex(policies).round(0).to_dict())
        print("\nPaired differences in net balance (Mm3), 95 % bootstrap interval over seasons")
        print(out[out.table == "paired_difference"].drop(columns="table").round(2).to_string(index=False))
        spread = d[d.train_seed >= 0].groupby(["scenario", "policy", "fold", "train_seed"]).net_balance_Mm3.mean().unstack()
        print("\nSpread between training seeds (max - min, mean over seasons, Mm3)")
        print(spread.max(axis=1).sub(spread.min(axis=1)).groupby(["scenario", "policy"]).mean().unstack().round(2).to_string())
        print(f"\nSummary written to {SUMMARY_FILE}")
    return out


# =====================================================================================================
#  6. Complete experimental protocol, data check, figures
# =====================================================================================================
SEEDS = CFG["evaluation"]["train_seeds"]
RUNS = ([(algo, algo, year, seed, 300) for seed in SEEDS for algo in CFG["training"]["algorithms"] for year in FOLDS]
        + [("ppo", "ppo600", year, seed, 600) for seed in SEEDS for year in FOLDS])


def run_experiments():
    todo = [r for r in RUNS if not (RUNS_DIR / run_name(r[1], r[2], r[3]) / "params.json").exists()]
    print(f"{len(RUNS)} training runs, {len(RUNS) - len(todo)} already done, {len(todo)} to run")
    for k, (algo, label, year, seed, iters) in enumerate(todo, 1):
        print(f"\n=== {k}/{len(todo)}: {label} | test season {year}/{str(year + 1)[2:]} | seed {seed} ===", flush=True)
        t0 = time.time()
        try:
            train(algo, year, seed, iterations=iters, label=label)
        except Exception:
            traceback.print_exc()
            print(f"Run {label} test {year} seed {seed} failed; moving on.", flush=True)
        print(f"({(time.time() - t0) / 60:.0f} min)")
    evaluate_all()
    summarise()


def check_environment(year=2011, scenario="central"):
    """Quick check of the data and the environment: runs the two baselines over one full season."""
    t0 = time.time()
    params = default_params()
    grid = Grid(season_file(year))
    print(f"Grid {grid.ny} x {grid.nx} | glaciogenic candidate cells: {grid.glaciogenic_area.sum()} | "
          f"hygroscopic candidate cells: {grid.hygroscopic_area.sum()}")
    season = [load_season(season_file(year), grid, params)]
    env = SeedingEnv(season, grid, scenario, params, mode="evaluate")
    print(f"Season {season_label(year)}: {len(season[0]['time'])} h | "
          f"glaciogenic opportunity in {season[0]['mask_g'].any(axis=(1, 2)).mean() * 100:.1f} % of hours | "
          f"hygroscopic in {season[0]['mask_h'].any(axis=(1, 2)).mean() * 100:.1f} % | "
          f"observation {env.obs_dim} values, {env.n_actions} actions, {len(env.agents)} aircraft")
    for policy in ["random", "greedy"]:
        st, _ = play_season(season, grid, scenario, 1, policy, params)
        print(f"  {policy:7s} gain {st['gain_Mm3']:7.3f} Mm3 | {st['flight_hours']:6.0f} flight hours | "
              f"{st['seedings']:5d} seedings | {st['reserved']:4d} reserved | net {st['net_balance_Mm3']:+.3f} Mm3")
    print(f"Done in {time.time() - t0:.0f} s")


def make_figures():
    """Figures of the paper (600 dpi PNG and vector PDF) in results/figures, with figure_numbers.csv."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.geodesic import Geodesic

    OUT = RESULTS_DIR / "figures"
    CACHE = OUT / "_cache"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    DPI = 600
    MAP_SCENARIO = "pessimistic"         # scenario of the decision logs used in Fig. 9 and 10

    YEARS = SEASON_ORDER                  # display order: wet to dry
    REGIME = {y: SEASONS[y] for y in YEARS}
    NAME = {y: season_label(y) for y in YEARS}
    ALGOS = ["mappo", "ippo", "ppo"]
    POLICIES = ["mappo", "ippo", "ppo", "greedy", "random"]
    POLICIES_NB = ["mappo", "ippo", "ppo", "ppo600", "greedy", "random"]
    TRAIN_SEEDS = tuple(CFG["evaluation"]["train_seeds"])
    SCENARIOS = CFG["evaluation"]["scenarios"]
    LABEL = {"mappo": "MAPPO", "ippo": "IPPO", "ppo": "PPO (centralised)", "ppo600": "PPO (600 iterations)",
             "greedy": "Greedy", "random": "Random"}
    SHORT = {"mappo": "MAPPO", "ippo": "IPPO", "ppo": "PPO", "ppo600": "PPO-600", "greedy": "Greedy", "random": "Random"}

    # ----------------------------- style -----------------------------
    # fixed categorical order, distinguishable for colour-vision deficiencies on adjacent pairs;
    # the non-learning baselines are grey so that they do not compete with the algorithms
    C = {"mappo": "#2a78d6", "ippo": "#eb6834", "ppo": "#1baf7a", "ppo600": "#0d6e4c", "greedy": "#52514e", "random": "#a9a8a3"}
    C_TECH = {"glaciogenic": "#6250d6", "hygroscopic": "#008300"}
    C_REGIME = {"wet": "#184f95", "normal": "#8a8985", "dry": "#c25a1a"}
    MK_REGIME = {"wet": "o", "normal": "s", "dry": "^"}
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    MM = 1 / 25.4                         # Elsevier widths: 90 mm (1 column), 140 mm (1.5), 190 mm (2)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": DPI, "pdf.fonttype": 42,
        "ps.fonttype": 42, "mathtext.default": "regular", "axes.titleweight": "bold", "axes.titlelocation": "center",
    })
    NUMBERS = []                          # plotted numbers, for cross-checking with the manuscript


    def panel_titles(fig):
        """Centred bold panel titles of the form '(a) Title'."""
        for ax in fig.axes:
            letter = getattr(ax, "_letter", None)
            if letter is None:
                continue
            t = (ax.get_title(loc="left") or ax.get_title(loc="center")).strip()
            ax.set_title("", loc="left"); ax.set_title("", loc="right")
            ax.set_title(f"({letter}) {t}" if t else f"({letter})", loc="center", fontsize=7.5, fontweight="bold",
                         color=INK, pad=5)


    def save(fig, name):
        panel_titles(fig)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close(fig)
        print(f"  {name}.png / .pdf")


    def letter(ax, txt):
        ax._letter = txt


    # ----------------------------- domain geometry -----------------------------
    params = default_params()
    grid = Grid(season_file(YEARS[0]))
    LAT, LON = grid.lat, grid.lon
    EXT = [LON.min() - 0.125, LON.max() + 0.125, LAT.min() - 0.125, LAT.max() + 0.125]
    BASES = params["bases"]

    PC = ccrs.PlateCarree()
    PROJ = ccrs.Mercator(central_longitude=-9.4)          # conformal; the ERA5 domain stays rectangular
    NE_LAND = cfeature.NaturalEarthFeature("physical", "land", "10m")
    NE_OCEAN = cfeature.NaturalEarthFeature("physical", "ocean", "10m")
    NE_COAST = cfeature.NaturalEarthFeature("physical", "coastline", "10m")
    LAND_COLOUR, OCEAN_COLOUR = "#f2efe8", "#e8eff5"
    FF = 8                                                 # oversampling of the 0.25-degree fields (cubic spline)
    LON_FF = np.linspace(LON.min(), LON.max(), len(LON) * FF)
    LAT_FF = np.linspace(LAT.max(), LAT.min(), len(LAT) * FF)
    LIGHT = mcolors.LightSource(azdeg=315, altdeg=35)
    OROG_FF = np.clip(zoom(np.clip(np.nan_to_num(grid.orog), 0, None), FF, order=3, mode="nearest"), 0, None)
    LAND_FF = zoom(gaussian_filter(grid.lsm.astype(float), 0.5), FF, order=1, mode="nearest") > 0.5


    def smooth_mask(area):
        """0.25-degree binary mask -> curved outline on the fine grid (no staircase)."""
        return zoom(gaussian_filter(area.astype(float), 0.55), FF, order=1, mode="nearest") >= 0.5


    def smooth_field(field, area=None, sigma=0.0):
        """0.25-degree field -> fine grid by cubic spline, restricted to an area with a smoothed edge."""
        a = np.nan_to_num(np.asarray(field, float), nan=0.0)
        if sigma:
            a = gaussian_filter(a, sigma)
        a = np.clip(zoom(a, FF, order=3, mode="nearest"), 0, None)
        return np.where(smooth_mask(area), a, np.nan) if area is not None else a


    def hillshade(ax, alpha=0.55, zorder=1.2):
        """Grey hillshading over land from the interpolated ERA5 orography."""
        h = LIGHT.hillshade(OROG_FF, vert_exag=0.08, dx=0.25 / FF * 111e3, dy=0.25 / FF * 111e3)
        flat = np.median(h[LAND_FF])                       # illumination of flat terrain
        shade = np.clip((flat - h) * 2.2, 0, 1)
        light = np.clip((h - flat) * 2.2, 0, 1)
        rgba = np.zeros((*h.shape, 4))
        rgba[..., :3] = np.where(shade[..., None] > 0, 0.18, 1.0)
        rgba[..., 3] = np.where(LAND_FF, np.maximum(shade, 0.6 * light) * alpha, 0)
        ax.imshow(rgba, extent=EXT, origin="upper", transform=PC, interpolation="bilinear", zorder=zorder,
                  regrid_shape=1600)


    def base_map(ax, relief=True):
        ax.set_extent(EXT, crs=PC)
        ax.set_rasterization_zorder(2.95)                  # smooth fields as 600-dpi images; lines and text stay vector
        ax.add_feature(NE_OCEAN, fc=OCEAN_COLOUR, ec="none", zorder=0)
        ax.add_feature(NE_LAND, fc=LAND_COLOUR, ec="none", zorder=0.5)
        if relief:
            hillshade(ax)


    def finish_map(ax, left=True, bottom=True, bases=False, atlas=False):
        """Natural Earth coastline, labelled graticule, thin frame; optional bases and glaciogenic outline."""
        ax.add_feature(NE_COAST, ec=INK, lw=0.55, fc="none", zorder=7)
        if atlas:
            ax.contour(LON_FF, LAT_FF, smooth_mask(grid.glaciogenic_area).astype(float), levels=[0.5], colors=INK2,
                       linewidths=0.45, transform=PC, zorder=7)
        gl = ax.gridlines(crs=PC, draw_labels=True, lw=0.35, color="#8f8d86", alpha=0.55, ls=(0, (2, 3)),
                          xlocs=np.arange(-16, 0, 4), ylocs=np.arange(22, 38, 4), x_inline=False, y_inline=False,
                          zorder=3)
        gl.top_labels = gl.right_labels = False
        gl.left_labels, gl.bottom_labels = left, bottom
        gl.xlabel_style = gl.ylabel_style = dict(size=6.2, color=INK2)
        gl.rotate_labels = False
        ax.spines["geo"].set_edgecolor(INK2); ax.spines["geo"].set_linewidth(0.6)
        if bases:
            for la, lo in BASES.values():
                ax.plot(lo, la, marker="*", ms=6.5, mfc="white", mec=INK, mew=0.6, transform=PC, zorder=9)


    def note(ax, txt):
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=INK, zorder=12,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc="white", ec=GRID, lw=0.4, alpha=0.9))


    def scale_bar(ax, lon0, lat0, km=200):
        dlon = km / (111.2 * np.cos(np.deg2rad(lat0)))
        for k, col in enumerate([INK, "white", INK, "white"]):
            ax.plot([lon0 + k * dlon / 4, lon0 + (k + 1) * dlon / 4], [lat0, lat0], color=col, lw=2.2,
                    solid_capstyle="butt", transform=PC, zorder=10)
        ax.plot([lon0, lon0 + dlon], [lat0, lat0], color=INK, lw=2.9, solid_capstyle="butt", transform=PC, zorder=9.5)
        ax.text(lon0 + dlon / 2, lat0 + 0.35, f"{km} km", ha="center", va="bottom", fontsize=6, transform=PC, zorder=10)


    def north_arrow(ax, x, y):
        ax.annotate("N", xy=(x, y + 0.07), xytext=(x, y), xycoords="axes fraction", ha="center", va="top", fontsize=7,
                    fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9), zorder=12)


    # =====================================================================================
    #  ERA5 aggregates (cached): opportunity frequency, seedable cells per hour, offshore IVT
    # =====================================================================================
    IVT_BINS = np.arange(0, 1305, 25)
    f_cache = CACHE / "opportunities.npz"
    if f_cache.exists():
        Z = dict(np.load(f_cache))
    else:
        print("Computing the seedability masks over the five seasons (once, a few minutes)...")
        Z = {}
        for y in YEARS:
            t0 = time.time()
            S = load_season(season_file(y), grid, params)
            Z[f"freq_g_{y}"] = S["mask_g"].mean(0)
            Z[f"freq_h_{y}"] = S["mask_h"].mean(0)
            Z[f"n_g_{y}"] = S["mask_g"].sum((1, 2))
            Z[f"n_h_{y}"] = S["mask_h"].sum((1, 2))
            Z[f"ivt_h_{y}"] = np.histogram(S["ivt"][:, grid.hygroscopic_area], bins=IVT_BINS)[0]
            Z[f"tp_land_{y}"] = (S["tp_mm"][:, grid.lsm]).sum(0).mean()
            del S
            print(f"  {NAME[y]}: {time.time() - t0:.0f} s")
        np.savez_compressed(f_cache, **Z)


    def composite(key, regime):
        return np.mean([Z[f"{key}_{y}"] for y in YEARS if REGIME[y] == regime], axis=0)


    # =====================================================================================
    #  Evaluation results and statistics (same method as in the manuscript)
    # =====================================================================================
    E = pd.read_csv(RESULTS_DIR / "evaluation_multiseed.csv")
    PER_SEASON = E.groupby(["scenario", "policy", "fold"]).mean(numeric_only=True).reset_index()
    RNG = np.random.default_rng(CFG["evaluation"]["bootstrap_seed"])
    IDX = RNG.integers(0, len(YEARS), size=(CFG["evaluation"]["bootstrap_resamples"], len(YEARS)))


    def bootstrap_ci(v):
        v = np.asarray(v, float)
        m = v[IDX].mean(1)
        return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


    def series(scenario, policy, col="net_balance_Mm3"):
        d = PER_SEASON[(PER_SEASON.scenario == scenario) & (PER_SEASON.policy == policy)].set_index("fold")
        return d.loc[YEARS, col].values


    def run_dir(policy, year, seed):
        return RUNS_DIR / run_name(policy, year, seed)


    # =====================================================================================
    #  Figure 1: study domain
    # =====================================================================================
    print("Figure 1")
    fig = plt.figure(figsize=(140 * MM, 150 * MM))
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    base_map(ax, relief=False)
    TERRAIN = mcolors.LinearSegmentedColormap.from_list(
        "terrain", ["#f4f1e6", "#e3d9bd", "#cdb88f", "#ae9165", "#856a47", "#5c4a31"])
    LEVELS_Z = [0, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]
    ax.contourf(LON_FF, LAT_FF, np.where(LAND_FF, OROG_FF, np.nan), levels=LEVELS_Z, cmap=TERRAIN,
                transform=PC, zorder=1, extend="max")
    hillshade(ax, alpha=0.6, zorder=1.3)
    g3, h3 = smooth_mask(grid.glaciogenic_area).astype(float), smooth_mask(grid.hygroscopic_area).astype(float)
    ax.contourf(LON_FF, LAT_FF, h3, levels=[0.5, 1.5], colors=[C_TECH["hygroscopic"]], alpha=0.22, transform=PC, zorder=2)
    ax.contour(LON_FF, LAT_FF, h3, levels=[0.5], colors=C_TECH["hygroscopic"], linewidths=0.8, linestyles="dashed",
               transform=PC, zorder=4)
    ax.contourf(LON_FF, LAT_FF, g3, levels=[0.5, 1.5], colors=[C_TECH["glaciogenic"]], alpha=0.38, transform=PC, zorder=3)
    ax.contour(LON_FF, LAT_FF, g3, levels=[0.5], colors=C_TECH["glaciogenic"], linewidths=0.9, transform=PC, zorder=4)
    geod = Geodesic()
    for la, lo in BASES.values():
        circle = np.asarray(geod.circle(lon=lo, lat=la, radius=params["speed_kmh"] * 1e3, n_samples=180))
        ax.plot(circle[:, 0], circle[:, 1], color=INK2, lw=0.5, ls=(0, (1.5, 2)), alpha=0.75, transform=PC, zorder=5)
    finish_map(ax, bases=True)
    OFFSET = {"Tangier": (0.35, -0.3), "Rabat": (0.3, 0.1), "Casablanca": (0.3, -0.3), "Marrakech": (0.3, -0.15),
              "Agadir": (-0.3, 0.0), "Dakhla": (0.3, 0.0)}
    for b, (la, lo) in BASES.items():
        dx, dy = OFFSET[b]
        ax.text(lo + dx, la + dy, b, fontsize=6.5, fontweight="bold", ha="left" if dx > 0 else "right",
                va="center", transform=PC, zorder=10,
                path_effects=[mpl.patheffects.withStroke(linewidth=2, foreground="white")])
    scale_bar(ax, -16.0, 22.4, 200)
    north_arrow(ax, 0.07, 0.30)
    ax.text(-13.6, 33.4, "Atlantic\nOcean", fontsize=7.5, style="italic", color="#3a5f86", ha="center", transform=PC)
    sm = mpl.cm.ScalarMappable(norm=mcolors.BoundaryNorm(LEVELS_Z, 256), cmap=TERRAIN)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.03, aspect=25)
    cb.set_label("Terrain height (m)"); cb.outline.set_linewidth(0.4)
    legend = [Patch(fc=mcolors.to_rgba(C_TECH["glaciogenic"], 0.38), ec=C_TECH["glaciogenic"], lw=0.9,
                    label=f"Glaciogenic candidate cells (terrain ≥ 1000 m, n = {int(grid.glaciogenic_area.sum())})"),
              Patch(fc=mcolors.to_rgba(C_TECH["hygroscopic"], 0.22), ec=C_TECH["hygroscopic"], ls="--", lw=0.8,
                    label=f"Hygroscopic candidate cells (ocean ≤ 200 km, n = {int(grid.hygroscopic_area.sum())})"),
              Line2D([], [], marker="*", ls="", ms=7, mfc="white", mec=INK, mew=0.6, label="Aircraft base (one aircraft each)"),
              Line2D([], [], color=INK2, lw=0.6, ls=(0, (1.5, 1.5)), label="One-hour flight range (350 km)")]
    ax.legend(handles=legend, loc="lower right", fontsize=5.8, handlelength=2.2, borderpad=0.6,
              frameon=True, framealpha=0.92, edgecolor=GRID)
    NUMBERS.extend([{"figure": "1", "quantity": "glaciogenic_cells", "value": int(grid.glaciogenic_area.sum())},
                {"figure": "1", "quantity": "hygroscopic_cells", "value": int(grid.hygroscopic_area.sum())},
                {"figure": "1", "quantity": "land_cells", "value": int(grid.lsm.sum())}])
    save(fig, "fig01_domain")

    # =====================================================================================
    #  Figure 2: share of hours with a seedable cell
    # =====================================================================================
    print("Figure 2")
    fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 156 * MM), subplot_kw=dict(projection=PROJ),
                            gridspec_kw=dict(wspace=0.04, hspace=0.12))
    titles = {"wet": "wet (2009/10, 2025/26)", "normal": "normal (2012/13)", "dry": "dry (2011/12, 2015/16)"}
    CM_G = mcolors.LinearSegmentedColormap.from_list("g3", ["#f3f1fc", "#d4cdf5", "#a99ce9", "#7c69dc", "#5541c4", "#352796", "#1f1661"])
    CM_H = mcolors.LinearSegmentedColormap.from_list("h3", ["#eef8f1", "#c8ead3", "#93d3a8", "#55b57b", "#23914f", "#0b6b35", "#04441f"])
    for r, (key, cmap, tech, area) in enumerate([("freq_g", CM_G, "Glaciogenic (G3)", grid.glaciogenic_area),
                                                  ("freq_h", CM_H, "Hygroscopic (H3)", grid.hygroscopic_area)]):
        vmax = np.ceil(max(composite(key, g)[area].max() for g in ("wet", "normal", "dry")) * 100 / 5) * 5
        levels = np.linspace(0, vmax, 21)
        for c, reg in enumerate(["wet", "normal", "dry"]):
            ax = axs[r, c]
            base_map(ax)
            raw = np.where(area, composite(key, reg) * 100, np.nan)
            f = smooth_field(np.where(area, raw, 0), area)
            im = ax.contourf(LON_FF, LAT_FF, f, levels=levels, cmap=cmap, extend="neither", transform=PC, zorder=2)
            ax.contour(LON_FF, LAT_FF, f, levels=levels[::4][1:], colors="white", linewidths=0.3, alpha=0.8,
                       transform=PC, zorder=2.5)
            finish_map(ax, left=(c == 0), bottom=(r == 1))
            ax.set_title(f"{'G3' if r == 0 else 'H3'}, {titles[reg]}", pad=4)
            letter(ax, "abcdef"[3 * r + c])
            mean = np.nanmean(raw)
            note(ax, f"mean {mean:.1f} %   max {np.nanmax(raw):.1f} %")
            NUMBERS.append({"figure": "2", "quantity": f"{key}_{reg}_mean_%", "value": round(mean, 2)})
        cb = fig.colorbar(im, ax=axs[r, :], shrink=0.88, pad=0.015, aspect=22, ticks=levels[::4])
        cb.set_label(f"{tech}: share of hours seedable (%)"); cb.outline.set_linewidth(0.4)
    save(fig, "fig02_opportunity_frequency")

    # =====================================================================================
    #  Figure 3: exceedance probabilities
    # =====================================================================================
    print("Figure 3")
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 64 * MM), gridspec_kw=dict(wspace=0.38))


    def exceedance(ax, x, p, reg, label):
        """Exceedance curve P(X >= x), filled down to the axis (log scale)."""
        ok = p > 0
        ax.fill_between(x[ok], 1e-7, p[ok], color=C_REGIME[reg], alpha=0.05, lw=0, zorder=1)
        ax.plot(x[ok], p[ok], color=C_REGIME[reg], lw=1.6, solid_capstyle="round", label=label, zorder=3)


    for k, (key, name) in enumerate([("n_g", "glaciogenic (G3)"), ("n_h", "hygroscopic (H3)")]):
        ax = axs[k]
        nmax = int(max(Z[f"{key}_{y}"].max() for y in YEARS))
        n = np.arange(1, nmax + 2)
        for reg in ["wet", "normal", "dry"]:
            v = np.concatenate([Z[f"{key}_{y}"] for y in YEARS if REGIME[y] == reg])
            none = (v == 0).mean() * 100
            exceedance(ax, n, np.array([(v >= m).mean() for m in n]), reg,
                       f"{reg.capitalize()} (≥ 1 cell in {100 - none:.0f} % of hours)")
            NUMBERS.append({"figure": "3", "quantity": f"{key}_{reg}_hours_without_opportunity_%", "value": round(none, 1)})
        ax.set_yscale("log"); ax.set_ylim(3e-4 if k == 0 else 1e-4, 1); ax.set_xlim(0, nmax * 1.02)
        ax.set_xlabel(f"Seedable {name} cells per hour, n")
        ax.set_ylabel("Exceedance probability P(N ≥ n)")
        ax.legend(loc="upper right", fontsize=5.8, handlelength=1.6)
        ax.grid(color=GRID, lw=0.4, which="major")
        ax.set_title(["Glaciogenic opportunities", "Hygroscopic opportunities"][k])
        letter(ax, "ab"[k])
    ax = axs[2]
    ax.axvspan(250, 1300, color="#f1efe9", lw=0, zorder=0)
    ax.text(1270, 0.6, "Atmospheric-river conditions\n(IVT ≥ 250 kg m$^{-1}$ s$^{-1}$)", fontsize=5.8, color=INK2,
            va="top", ha="right")
    for reg in ["wet", "normal", "dry"]:
        h = np.sum([Z[f"ivt_h_{y}"] for y in YEARS if REGIME[y] == reg], axis=0).astype(float)
        exceedance(ax, IVT_BINS[:-1], h[::-1].cumsum()[::-1] / h.sum(), reg, reg.capitalize())
        NUMBERS.append({"figure": "3", "quantity": f"offshore_ivt_{reg}_%_above_250",
                        "value": round(h[IVT_BINS[:-1] >= 250].sum() / h.sum() * 100, 2)})
    ax.axvline(250, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_yscale("log"); ax.set_xlim(0, 1300); ax.set_ylim(1e-6, 1.5)
    ax.set_xlabel("Offshore IVT (kg m$^{-1}$ s$^{-1}$)")
    ax.set_ylabel("Exceedance probability")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.05), fontsize=6)
    ax.grid(color=GRID, lw=0.4)
    ax.set_title("Offshore moisture transport")
    letter(ax, "c")
    save(fig, "fig03_opportunity_exceedance")

    # Figure 4 (architecture) is drawn in docs/architecture.drawio.

    # =====================================================================================
    #  Figure 5: learning curves
    # =====================================================================================
    print("Figure 5")
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), gridspec_kw=dict(wspace=0.25))
    for a in ALGOS + ["ppo600"]:
        curves = {"mean_episode_return_Mm3": [], "mean_episode_gain_Mm3": []}
        for y in YEARS:
            for g in TRAIN_SEEDS:
                f = run_dir(a, y, g) / "learning_curve.csv"
                if f.exists():
                    d = pd.read_csv(f)
                    for k in curves:
                        curves[k].append(d[k].rolling(10, min_periods=1).mean().values)
        for ax, k in zip(axs, curves):
            if not curves[k]:
                continue
            L = min(map(len, curves[k]))
            A = np.array([c[:L] for c in curves[k]])
            it = np.arange(1, L + 1)
            m, s = A.mean(0), A.std(0)
            if a != "ppo600":
                ax.fill_between(it, m - s, m + s, color=C[a], alpha=0.15, lw=0)
            ax.plot(it, m, color=C[a], lw=1.0 if a == "ppo600" else 1.2, ls=(0, (4, 2)) if a == "ppo600" else "-",
                    label=f"{LABEL[a]} (n = {len(A)})")
    for ax, yl, l in zip(axs, ["Team return per 72-h episode (Mm$^3$)", "Water gain per 72-h episode (Mm$^3$)"], "ab"):
        ax.set_xlabel("Training iteration (16 episodes each)"); ax.set_ylabel(yl)
        ax.grid(color=GRID, lw=0.4); ax.set_xlim(1, 600); ax.axvline(300, color=INK2, lw=0.5, ls=":")
        ax.set_title({"a": "Team return", "b": "Gross water gain"}[l]); letter(ax, l)
    axs[0].legend(loc="lower right")
    axs[1].text(0.98, 0.04, "pessimistic scenario; mean ± 1 s.d. over 5 folds × 3 seeds\n"
                "(10-iteration moving mean; PPO-600: mean only)",
                transform=axs[1].transAxes, ha="right", va="bottom", fontsize=5.8, color=INK2)
    save(fig, "fig05_learning_curves")

    # =====================================================================================
    #  Figure 6: net water balance by policy and scenario
    # =====================================================================================
    print("Figure 6")
    mpl.rcParams["hatch.color"] = "white"; mpl.rcParams["hatch.linewidth"] = 0.6
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 64 * MM), gridspec_kw=dict(wspace=0.28))
    for ax, sc, l in zip(axs, SCENARIOS, "abc"):
        for k, p in enumerate(POLICIES_NB):
            v = series(sc, p)
            m, lo, hi = bootstrap_ci(v)
            ax.bar(k, m, width=0.68, color=C[p], ec="white", lw=0.8, zorder=2, hatch="////" if p == "ppo600" else None)
            ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.8, capsize=2.2, capthick=0.8, zorder=4)
            for j, y in enumerate(YEARS):
                ax.plot(k + (j - 2) * 0.08, v[j], MK_REGIME[REGIME[y]], ms=2.8, mfc="white", mec=INK2, mew=0.5, zorder=5)
            NUMBERS.append({"figure": "6", "quantity": f"net_balance_{sc}_{p}", "value": round(m, 3),
                            "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.set_xticks(range(len(POLICIES_NB)))
        ax.set_xticklabels([SHORT[p] for p in POLICIES_NB], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(f"{sc.capitalize()} effect scenario", pad=5)
        ax.set_ylabel("Net water balance (Mm$^3$ per season)" if sc == SCENARIOS[0] else "")
        ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        letter(ax, l)
    legend = [Line2D([], [], marker=MK_REGIME[r], ls="", ms=3.5, mfc="white", mec=INK2, mew=0.6, label=f"{r} season")
              for r in ("wet", "normal", "dry")] + [Line2D([], [], color=INK, lw=0.8, label="95 % bootstrap CI (5 seasons)")]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig06_net_balance")

    # =====================================================================================
    #  Figure 7: paired differences
    # =====================================================================================
    print("Figure 7")
    pairs = [("mappo", "ippo"), ("mappo", "ppo"), ("mappo", "ppo600"), ("ippo", "ppo"), ("ppo600", "ppo"),
             ("mappo", "greedy"), ("mappo", "random"), ("ippo", "greedy"), ("ppo", "greedy")]
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 82 * MM), sharey=True, gridspec_kw=dict(wspace=0.22))
    for ax, sc, l in zip(axs, SCENARIOS, "abc"):
        for k, (a, b) in enumerate(pairs):
            d = series(sc, a) - series(sc, b)
            m, lo, hi = bootstrap_ci(d)
            won = int((d > 0).sum())
            sig = lo > 0 or hi < 0
            yk = len(pairs) - 1 - k
            ax.plot([lo, hi], [yk, yk], color=C[a], lw=1.6 if sig else 1.0, solid_capstyle="round", zorder=3)
            ax.plot(m, yk, "D", ms=4.2, mfc=C[a] if sig else "white", mec=C[a], mew=1.0, zorder=4)
            ax.text(1.03, yk, f"{won}/5", transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    ha="left", va="center", fontsize=6, color=INK if sig else INK2, fontweight="bold" if sig else "normal")
            NUMBERS.append({"figure": "7", "quantity": f"diff_{sc}_{a}-{b}", "value": round(m, 3), "ci_low": round(lo, 3),
                            "ci_high": round(hi, 3), "seasons_won": won})
        ax.axvline(0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_title(f"{sc.capitalize()} scenario", pad=5)
        ax.set_xlabel("Paired difference in net balance (Mm$^3$)")
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.spines["left"].set_visible(sc == SCENARIOS[0])
        ax.tick_params(axis="y", left=(sc == SCENARIOS[0]))
        letter(ax, l)
    axs[0].set_yticks(range(len(pairs)))
    axs[0].set_yticklabels([f"{SHORT[a]} − {SHORT[b]}" for a, b in pairs][::-1])
    legend = [Line2D([], [], marker="D", color=INK2, ms=4, mfc=INK2, lw=1.6, label="95 % CI excludes 0"),
              Line2D([], [], marker="D", color=INK2, ms=4, mfc="white", lw=1.0, label="95 % CI includes 0"),
              Line2D([], [], ls="", label="n/5 = held-out seasons won")]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    save(fig, "fig07_paired_differences")

    # =====================================================================================
    #  Figure 8: fleet behaviour by season
    # =====================================================================================
    print("Figure 8")
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 64 * MM), gridspec_kw=dict(wspace=0.22))
    w = 0.16
    for ax, col, yl, l in [(axs[0], "flight_hours", "Flight hours per season", "a"),
                           (axs[1], "glaciogenic_share", "Glaciogenic share of seedings", "b")]:
        for k, p in enumerate(POLICIES):
            v = series(SCENARIOS[0], p, col)
            ax.bar(np.arange(5) + (k - 2) * w, v, width=w * 0.92, color=C[p], ec="white", lw=0.4, label=LABEL[p], zorder=2)
            for j, y in enumerate(YEARS):
                NUMBERS.append({"figure": "8", "quantity": f"{col}_{y}_{p}", "value": round(float(v[j]), 3)})
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{NAME[y]}\n({REGIME[y]})" for y in YEARS])
        ax.set_ylabel(yl); ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
        ax.set_title({"a": "Flight effort", "b": "Choice of technique"}[l]); letter(ax, l)
    axs[1].set_ylim(0, 1); axs[1].axhline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig08_fleet_by_season")

    # =====================================================================================
    #  Figure 9: seeding density maps
    # =====================================================================================
    print("Figure 9")
    MAP_SEASONS = [2009, 2011]


    def greedy_log(y):
        f = CACHE / f"greedy_decisions_{y}_{MAP_SCENARIO}.csv"
        if f.exists():
            return pd.read_csv(f)
        print(f"  replaying the greedy rule over {NAME[y]} (once)...")
        S = [load_season(season_file(y), grid, params)]
        _, log = play_season(S, grid, MAP_SCENARIO, 1, "greedy", params)
        log.to_csv(f, index=False)
        return log


    def density(log):
        log = log[~log["reserved"].astype(str).str.lower().eq("true")]
        D = np.zeros(grid.lsm.shape)
        np.add.at(D, (log["i"].astype(int).values, log["j"].astype(int).values), 1)
        return D, (log["technique"] == "glaciogenic").mean() if len(log) else np.nan, len(log)


    cols = ["mappo", "ippo", "ppo", "greedy"]
    fig, axs = plt.subplots(2, 4, figsize=(190 * MM, 118 * MM), subplot_kw=dict(projection=PROJ),
                            gridspec_kw=dict(wspace=0.04, hspace=0.12))
    CM_D = mcolors.LinearSegmentedColormap.from_list(
        "density", ["#fff3c4", "#fdd27a", "#f9a04b", "#eb6834", "#c63a2c", "#8e1d2c", "#4f0f2a"])
    LEVELS_D = np.geomspace(0.5, 200, 17)
    norm = mcolors.BoundaryNorm(LEVELS_D, CM_D.N, extend="max")
    for r, y in enumerate(MAP_SEASONS):
        for c, p in enumerate(cols):
            ax = axs[r, c]
            if p == "greedy":
                log = greedy_log(y)
            else:
                f = run_dir(p, y, 0) / f"decisions_{MAP_SCENARIO}.csv"
                log = pd.read_csv(f) if f.exists() else None
            base_map(ax)
            if log is not None and len(log):
                D, share_g, n = density(log)
                Dl = smooth_field(D, sigma=0.8)                 # Gaussian kernel, sigma of about 20 km
                Dl = np.where(Dl >= LEVELS_D[0], Dl, np.nan)
                ax.contourf(LON_FF, LAT_FF, Dl, levels=LEVELS_D, cmap=CM_D, norm=norm, extend="max", transform=PC,
                            zorder=2, alpha=0.95)
                note(ax, f"{n} seedings\nglaciogenic {share_g * 100:.0f} %")
                NUMBERS.append({"figure": "9", "quantity": f"seedings_{y}_{p}", "value": n,
                                "glaciogenic_share": round(float(share_g), 3)})
            else:
                ax.text(0.5, 0.5, "log not available", transform=ax.transAxes, ha="center", fontsize=6, color=INK2)
            finish_map(ax, left=(c == 0), bottom=(r == 1), bases=True, atlas=True)
            ax.set_title(LABEL[p], pad=4)
            if c == 0:
                ax.text(-0.30, 0.5, f"{NAME[y]} ({REGIME[y]})", transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=7, fontweight="bold")
            letter(ax, "abcdefgh"[4 * r + c])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CM_D), ax=axs, shrink=0.75, pad=0.012, aspect=28,
                      ticks=[0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    cb.ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100", "200"])
    cb.set_label(f"Seeding density per cell and season (kernel-smoothed, {MAP_SCENARIO} scenario)")
    cb.outline.set_linewidth(0.4)
    save(fig, "fig09_seeding_density")

    # =====================================================================================
    #  Figure 10: randomised reserve
    # =====================================================================================
    print("Figure 10")
    fig, axs = plt.subplots(2, 2, figsize=(190 * MM, 92 * MM),
                            gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08, wspace=0.25, width_ratios=[1.6, 1]))
    gs = axs[0, 1].get_gridspec(); axs[0, 1].remove(); axs[1, 1].remove()
    axr = fig.add_subplot(gs[:, 1])
    for a in ALGOS:
        scores = []
        for y in YEARS:
            for g in TRAIN_SEEDS:
                f = run_dir(a, y, g) / f"decisions_{MAP_SCENARIO}.csv"
                if f.exists():
                    scores.append(pd.read_csv(f, usecols=["score"])["score"].values)
        if scores:
            v = np.concatenate(scores)
            xs = np.linspace(0, 1, 400)
            dk = 3 * gaussian_kde(np.concatenate([v, -v, 2 - v]), bw_method=0.04)(xs)   # reflection at 0 and 1
            axs[0, 0].fill_between(xs, 0, dk, color=C[a], alpha=0.12, lw=0)
            axs[0, 0].plot(xs, dk, color=C[a], lw=1.6, label=f"{LABEL[a]} (median s = {np.median(v):.2f})")
            axs[0, 0].axvline(np.median(v), color=C[a], lw=0.7, ls=(0, (2, 2)))
            NUMBERS.append({"figure": "10", "quantity": f"median_score_{a}", "value": round(float(np.median(v)), 3)})
    axs[0, 0].set_ylabel("Density of selected targets"); axs[0, 0].set_xticklabels([])
    axs[0, 0].set_xlim(0, 1); axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].legend(loc="upper left"); axs[0, 0].grid(color=GRID, lw=0.4)
    axs[0, 0].set_title("Selected targets and reserve probability"); letter(axs[0, 0], "a")
    r = params["reserve"]
    s = np.linspace(0, 1, 101)
    p_res = r["r_max"] - (r["r_max"] - r["r_min"]) * s
    axs[1, 0].plot(s, p_res, color=INK, lw=1.3)
    axs[1, 0].fill_between(s, 0, p_res, color="#d9d8d3", lw=0)
    axs[1, 0].set_ylim(0, 0.55); axs[1, 0].set_xlim(0, 1)
    axs[1, 0].set_xlabel("Seedability score s"); axs[1, 0].set_ylabel("π$_{res}$(s)")
    axs[1, 0].text(0.04, 0.06, "π$_{res}$(s) = 0.50 − 0.40 s,  0.10 ≤ π$_{res}$ ≤ 0.50", fontsize=6, color=INK)
    axs[1, 0].grid(color=GRID, lw=0.4)
    for k, p in enumerate(POLICIES):
        d = E[(E.scenario == SCENARIOS[0]) & (E.policy == p)]
        share = (d.reserved / (d.reserved + d.seedings)).groupby(d.fold).mean().loc[YEARS].values
        m, lo, hi = bootstrap_ci(share * 100)
        yk = len(POLICIES) - 1 - k
        axr.barh(yk, m, color=C[p], height=0.62, ec="white", zorder=2)
        axr.errorbar(m, yk, xerr=[[m - lo], [hi - m]], color=INK, lw=0.8, capsize=2, zorder=3)
        axr.text(hi + 0.6, yk, f"{m:.1f} %", va="center", fontsize=6)
        NUMBERS.append({"figure": "10", "quantity": f"reserve_share_{p}_%", "value": round(m, 2),
                        "ci_low": round(lo, 2), "ci_high": round(hi, 2)})
    axr.set_yticks(range(5)); axr.set_yticklabels([SHORT[p] for p in POLICIES][::-1])
    axr.set_xlabel("Decisions withheld by the reserve (%)"); axr.set_xlim(0, 35)
    axr.grid(axis="x", color=GRID, lw=0.4, zorder=0); axr.set_title("Decisions withheld"); letter(axr, "b")
    save(fig, "fig10_randomised_reserve")

    # =====================================================================================
    #  Figure S1: reproducibility across training seeds
    # =====================================================================================
    print("Figure S1")
    S1 = E[E.policy.isin(ALGOS) & E.train_seed.isin(TRAIN_SEEDS)].groupby(
        ["scenario", "policy", "fold", "train_seed"]).net_balance_Mm3.mean().unstack()
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 64 * MM), gridspec_kw=dict(wspace=0.28))
    MK_SEED = {0: "o", 1: "s", 2: "^"}
    for ax, sc, l in zip(axs, SCENARIOS, "abc"):
        d = S1.loc[sc]
        for ia, a in enumerate(ALGOS):
            for j, y in enumerate(YEARS):
                v = d.loc[(a, y)].dropna()
                x = j + (ia - 1) * 0.24
                ax.plot([x, x], [v.min(), v.max()], color=C[a], lw=1.6, solid_capstyle="round", alpha=0.55, zorder=2)
                for g, val in v.items():
                    ax.plot(x, val, MK_SEED[int(g)], ms=3.4, mfc=C[a], mec="white", mew=0.4, zorder=3)
            spread = (d.loc[a].max(axis=1) - d.loc[a].min(axis=1)).mean()
            NUMBERS.append({"figure": "S1", "quantity": f"seed_spread_{sc}_{a}", "value": round(float(spread), 3)})
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{NAME[y]}\n({REGIME[y]})" for y in YEARS], fontsize=5.8)
        ax.set_ylabel("Net water balance (Mm$^3$ per season)" if sc == SCENARIOS[0] else "")
        ax.set_title(f"{sc.capitalize()} scenario", pad=5); ax.grid(axis="y", color=GRID, lw=0.4); letter(ax, l)
    legend = [Patch(fc=C[a], label=LABEL[a]) for a in ALGOS] + \
             [Line2D([], [], marker=MK_SEED[g], ls="", ms=4, mfc=INK2, mec="white", label=f"training seed {g}")
              for g in TRAIN_SEEDS]
    fig.legend(handles=legend, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1))
    save(fig, "figS1_seed_reproducibility")

    pd.DataFrame(NUMBERS).to_csv(OUT / "figure_numbers.csv", index=False)
    print(f"Done: figures and figure_numbers.csv in {OUT}")


# =====================================================================================================
#  7. Command line / Jupyter entry point
# =====================================================================================================
# In Jupyter, choose the task here and run the cell:
#   "check", "train", "experiments", "evaluate", "summary" or "figures"
TASK = "summary"
TRAIN_OPTIONS = {"algo": "mappo", "test_year": 2009, "seed": 0, "iterations": None, "label": None}
CHECK_OPTIONS = {"year": 2011, "scenario": "central"}


def run(task, **options):
    if task == "check":
        check_environment(**{**CHECK_OPTIONS, **options})
    elif task == "train":
        train(**{**TRAIN_OPTIONS, **options})
    elif task == "experiments":
        run_experiments()
    elif task == "evaluate":
        evaluate_all()
        summarise()
    elif task == "summary":
        summarise()
    elif task == "figures":
        make_figures()
    else:
        raise ValueError(f"unknown task: {task}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    p = sub.add_parser("check", help="run the two baselines over one season")
    p.add_argument("--year", type=int, default=2011, choices=sorted(SEASONS))
    p.add_argument("--scenario", default="central")
    p = sub.add_parser("train", help="train one policy on one fold")
    p.add_argument("--algo", choices=ALGORITHMS, default="mappo")
    p.add_argument("--test-year", type=int, choices=sorted(SEASONS), default=2012,
                   help="first year of the held-out season (e.g. 2009 for 2009/10)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--label", default=None, help="policy name used in the outputs (default: the algorithm)")
    sub.add_parser("experiments", help="complete protocol of the paper (resumable)")
    sub.add_parser("evaluate", help="paired evaluation of all completed runs, then summary")
    sub.add_parser("summary", help="summary statistics of results/evaluation_multiseed.csv")
    sub.add_parser("figures", help="figures of the paper")
    a = ap.parse_args()
    if a.task == "check":
        run("check", year=a.year, scenario=a.scenario)
    elif a.task == "train":
        run("train", algo=a.algo, test_year=a.test_year, seed=a.seed, iterations=a.iterations, label=a.label)
    else:
        run(a.task)


if __name__ == "__main__":
    if "ipykernel" in sys.modules:          # Jupyter cell
        run(TASK)
    else:
        main()
