#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
w90_harmonic_interband_ablation_v15.py

New in v15
----------
✅ Step A/B/C extensions to support the "hopping 灵敏度 → 物理调控" workflow:

  - Step A: Build single-hopping (R,i,j) Hermitian-merged knob sensitivities
            S_intra, S_inter, S_total = ∂C/∂λ  (fast, analytic, at chosen k0/direction).

  - Step B: Optional toy knob experiment: scale selected (R,i,j) hoppings by λ and recompute
            curvature with re-diagonalization (nonlinear response check).

  - Step C (P0): Optional two-HR mapping/validation:
            given HR_FILE_P0 (e.g. strained), fit λ_j between HR(0) and HR(p) for each knob,
            predict ΔC ≈ Σ_j S_j Δλ_j, and compare with the "real" ΔC computed from the full HR(p).

✅ v14 feature kept: you can force include higher harmonic groups (e.g. group 3/4/5) via
   ABLATE_GROUP_MODE = "upto" and ABLATE_GROUP_MAX = 5.

What is new vs v5 (v6 update)
-----------------
v3 already answered: "Which *real-space* R (or harmonic group) drives band curvature change?"
v4 adds the next missing resolution you requested:

  ✅ Within the same real-space direction / same R-group, hr.dat contains a full matrix H_ij(R)
     between different Wannier orbitals (i,j). v4 resolves curvature-driving terms down to
     (R, i, j) or (group, i, j).

So you can answer questions like:
  - For group |R2|=1 (Γ→Y first harmonic), which *orbital-to-orbital* hoppings dominate C_intra?
  - Which orbital pairs mainly control the *interband* term through velocity couplings?

Mathematical notes (why this is possible)
-----------------------------------------
1) Intraband term is linear in H(R), therefore it can be decomposed exactly (at fixed eigenvector |n>):
      C_intra = Σ_R Re[ <n| D2_R |n> ]
      D2_R = -(dotR^2) * exp(i2π k·R)/deg * H(R)
   and further into orbital pairs (i,j) because
      <n|D2_R|n> = Σ_{i,j} n_i* [D2_R]_{ij} n_j

2) Interband term is quadratic in V_nm = <n|D1|m>, so a unique "per-(i,j)" decomposition of |V|^2
   is not strictly additive. Instead v4 provides a *coherent linear-response sensitivity* (first order)
   for each (group,i,j), computed at baseline eigenvectors:

      Sens_inter(group,i,j) = dC_inter/dλ |_{λ=1}
                            = 4 Σ_{m≠n} Re[ V_nm* v_nm^{(group,i,j)} ] / (E_n - E_m)

   where v_nm^{(group,i,j)} is the partial amplitude of V_nm coming from that (group,i,j) subset.
   This is the cleanest way to "attribute" interband curvature to orbital hoppings without doing
   thousands of finite-difference ablations.

Gauge note (important)
----------------------
Orbital-resolved results depend on the Wannier gauge / ordering. For strain comparison,
you should keep the same projections / disentanglement windows so the Wannier functions track
the same physics across structures.

Outputs (new in v4)
-------------------
- orbpair_group_<g>_ranking.csv
    Top orbital pairs (i,j) within each selected group g, with:
      intra_contrib_ij   (exact linear decomposition of C_intra at fixed |n>)
      inter_sens_ij      (coherent sensitivity dC_inter/dλ at baseline)
      total_sens_ij      (intra_contrib + inter_sens)

- orbpair_by_R_top.csv   (optional)
    For each exact R vector, prints the top orbital pairs contributing to C_intra at that R.

Other v3 outputs are kept (baseline interband table, intra_contrib_by_R, group scores, ablation ranking, etc.)

"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# =============================================================================
# USER PARAMETERS（请在这里直接修改；推荐工作流）
# =============================================================================
# --- Input files ---
HR_FILE = "wannier90_hr.dat"
POSCAR_FILE = "POSCAR"

# --- Lattice source for derivative units (IMPORTANT for curvature) ---
# Curvature uses real-space distances R_abs in Å (from lattice vectors).
# If your POSCAR lattice vectors are not identical to the unit cell used to generate
# wannier90_hr.dat / seedname_band.dat, energies will still match but curvatures WILL be wrong.
#
# Options:
#   'auto'  : prefer unit_cell_cart from <seedname>.win if available; else POSCAR
#   'win'   : force using unit_cell_cart from .win (recommended if you have it)
#   'poscar': force using POSCAR lattice vectors
LATTICE_SOURCE = 'auto'

# --- k point (fractional reciprocal coords) ---
K_FRAC = (0.0, 0.0, 0.0)

# --- Target bands (1-based within the Wannier TB subspace at the chosen k) ---
BAND_N = 17
BAND_M = 16  # only for convenience print; interband is always a sum over all m≠n

# --- Direction for curvature derivative ---
# "cart" uses DIR_CART directly (Cartesian reciprocal direction, normalized).
# "from_kline" uses KLINE_START -> KLINE_END and POSCAR reciprocal vectors to define direction.
DIR_MODE = "cart"
DIR_CART = (0.0, 1.0, 0.0)

# --- k-line definition (also used for harmonic grouping n(R)=q·R) ---
KLINE_START = (0.0, 0.0, 0.0)
KLINE_END   = (0.0, 0.5, 0.0)


# --- NEW v13: choose the expansion point k0 along a k-line by minimizing |v| ---
# A visually "flat" band segment is not guaranteed to have v≈0 at Γ.
# This block scans the segment and finds k* where |v| is minimal for BAND_N,
# where v = dE/dk_u = <n|dH/dk_u|n> along the chosen direction u_hat.
#
# Typical use (Γ→Y):
#   DIR_MODE   = "from_kline"
#   KLINE_START= (0,0,0)
#   KLINE_END  = (0,0.5,0)
#   AUTO_K0_MINABS_V_ENABLE = True
#   AUTO_K0_MINABS_V_USE_FOR_ANALYSIS = True   (recommended)
AUTO_K0_MINABS_V_ENABLE = True

# Scan segment in fractional reciprocal coordinates.
# If None: use KLINE_START / KLINE_END
KSCAN_START = None
KSCAN_END   = None

# Number of scan points along the segment.
#   "win": use bands_num_points from <seedname>.win (if available) to match wannier90 band sampling
#   int  : explicit number (>=3)
KSCAN_NUM_POINTS: Union[str, int] = "win"

# When searching for the minimum, ignore endpoints (t=0 and t=1) to avoid boundary artefacts.
KSCAN_EXCLUDE_ENDPOINTS = True

# Track the band continuously by overlap (recommended if there are nearby crossings).
KSCAN_TRACK_BY_OVERLAP = True

# If v changes sign between sampled points, refine the minimum using bisection to locate v≈0.
KSCAN_REFINE_ROOT = True
KSCAN_ROOT_MAX_ITERS = 80
KSCAN_ROOT_TOL_T = 1e-12   # tolerance in the line parameter t (0..1)

# Export scan table (k, E, v, |v| vs t)
EXPORT_KSCAN_TABLE = True
KSCAN_TABLE_CSV = "kline_scan_minabs_v.csv"

# If True, use k* (min |v| point) as the k0 for ALL subsequent analyses (curvature, hopping, ablation, etc.)
AUTO_K0_MINABS_V_USE_FOR_ANALYSIS = True

# --- Harmonic grouping controls ---
DENOM_MAX = 24
MAX_ABS_N = None  # e.g. 6 to merge very large |n| into a single ">6" group label

# --- Reporting / exports ---
TOPN_BANDS = 8
EXPORT_FULL_INTERBAND_TABLE = True
EXPORT_INTRA_PER_R = True

# --- Group pre-screen + ablation ---
# v14 adds a more flexible ablation group selection.
#
# ABLATE_GROUP_MODE:
#   "top"  : ablate TOP_GROUPS_FOR_ABLATION groups ranked by intraband abs_sum (old behavior)
#   "upto" : ablate all integer groups g=0..ABLATE_GROUP_MAX (inclusive)
#   "list" : ablate exactly the groups in ABLATE_GROUP_LIST (integers and/or strings)
#   "all"  : ablate all groups present in the current HR file
#
# NOTE:
#   - If you set MAX_ABS_N to a small number, large-|n| groups may be merged into a string label like ">6".
#     In that case, groups above MAX_ABS_N no longer exist as separate integers.
#   - If your HR comes from a coarse mp_grid along the chosen direction, the largest |n| present may be small
#     (e.g. only groups 0,1,2). To obtain groups 3,4,5 you need a denser mp_grid when generating Wannier90 HR.
TOP_GROUPS_FOR_ABLATION = 8
ABLATE_GROUP_MODE = "upto"       # "top" / "upto" / "list" / "all"
ABLATE_GROUP_MAX = 5             # used when ABLATE_GROUP_MODE="upto"
ABLATE_GROUP_LIST = [0, 1, 2, 3, 4, 5]  # used when ABLATE_GROUP_MODE="list"
LAMBDA_ABLATE = 0.95
MIN_OVERLAP_WARN = 0.90

# --- NEW: Orbital-resolved (Wannier-orbital pair) analysis ---
EXPORT_ORBPAIR_GROUP_RANKING = True   # write orbpair_group_<g>_ranking.csv
TOP_ORBPAIRS_PER_GROUP = 40           # number of orbital pairs to keep in each group ranking
ORBPAIR_RANK_SORT = "abs_total"       # "abs_total" / "abs_intra" / "abs_inter"

# Export exact-R orbital-pair contributions for C_intra
# If EXPORT_ORBPAIR_BY_R_TOP=True, we write one file "orbpair_by_R_top.csv"
# containing (for each R) the top TOP_ORBPAIRS_PER_R pairs by |contrib|.
EXPORT_ORBPAIR_BY_R_TOP = True
TOP_ORBPAIRS_PER_R = 20

# --- NEW: Post-process orbpair_by_R_top.csv to remove Hermitian-equivalent duplicates ---
# In orbpair_by_R_top.csv you may see pairs of rows like:
#   (R, i, j, label_i, label_j)  and  (-R, j, i, label_j, label_i)
# which are Hermitian-conjugate representations of the same physical hopping parameter.
# This post-processing merges them into one canonical row to make ranking/reading clearer.
MERGE_ORBPAIR_BY_R_TOP_HERMITIAN = True

# Canonical choice for equivalent pair:
#   - If |dotR_Ang| > MERGE_ORBPAIR_DOTR_EPS: enforce dotR_Ang >= 0 (i.e. keep the 'forward' direction)
#   - Else: fall back to a deterministic lexicographic rule.
MERGE_ORBPAIR_CANONICAL_USE_DOTR = True
MERGE_ORBPAIR_DOTR_EPS = 1e-10

# If you also want to merge across equivalent atoms (e.g. Ga1_px and Ga3_px -> Ga_px), set this True.
# WARNING: this is only safe if those atoms are symmetry-equivalent in your structure.
EXPORT_ORBPAIR_BY_R_TOP_MERGED_SHORTLABEL = False

# Output filenames for merged tables
ORBPAIR_BY_R_TOP_MERGED_FILE = "orbpair_by_R_top_herm_merged.csv"
ORBPAIR_BY_R_TOP_MERGED_SHORTLABEL_FILE = "orbpair_by_R_top_herm_merged_shortlabel.csv"


# --- NEW v5: Wannier-orbital labels (auto from .win/.wout) ---
# For orbital-resolved outputs (orbpair rankings) it is very helpful to have human-readable labels
# for each Wannier function (WF index). You can still set WANNIER_LABELS manually, but by default
# v5 will try to build labels automatically from:
#   (1) seedname.win   (begin projections ... end projections) + POSCAR atom list
#   (2) seedname.wout  (if it contains a clear "projection / trial orbital" table; used as fallback)
#
# Recommended: keep these files in the same folder as HR_FILE, with the same seedname:
#   seedname_hr.dat, seedname.win, seedname.wout
AUTO_WANNIER_LABELS = True     # if True and WANNIER_LABELS is None, attempt automatic labeling
SEEDNAME = None                # optional; if None, inferred from HR_FILE name (seedname_hr.dat -> seedname)
WIN_FILE = None                # optional; if None, uses "<seedname>.win" if present
WOUT_FILE = None               # optional; if None, uses "<seedname>.wout" if present
EXPORT_WANNIER_LABELS = True   # write a "wannier_labels_used.csv" file so you can verify mapping

# Manual override (highest priority):
# If you already know the exact order of your Wannier functions, set this list (length=num_wann)
# Example:
#   WANNIER_LABELS = ["Fe1_dxy", "Fe1_dyz", ..., "O3_pz", ...]
WANNIER_LABELS: Optional[List[str]] = None


# Interband sensitivity scope:
#   INTER_SENS_M_LIST = []    -> sum over all m≠n (recommended)
#   INTER_SENS_M_LIST = [19]  -> only sensitivity coming from coupling to band m=19
INTER_SENS_M_LIST: List[int] = []

# --- NEW v10: Numerical curvature self-check from Wannier90 band file (seedname_band.dat) ---
# This compares the analytic curvature (from hr.dat) at your chosen k point and direction
# against a finite-difference curvature extracted directly from the band-plot file.
#
# Why it helps:
#   - If your DIR_CART is not aligned with the band-plot path direction (e.g., Γ→Y),
#     analytic and numerical curvatures will differ. This block will detect that and warn.
#
# How it works:
#   - Reads BAND_DAT_FILE (two columns: k-distance and energy) for BAND_CHECK_BAND_INDEX.
#   - Locates the target k-distance (KD_TARGET) and computes one-sided quadratic curvature
#     on the left and right segments:
#         C_left  from points (i-2,i-1,i)
#         C_right from points (i,i+1,i+2)
#   - Compares your analytic C_total to both, reports the closer one and warns if mismatch is large.
#
# Recommended:
#   - Set DIR_MODE="from_kline" with KLINE_START/KLINE_END matching the segment you want to compare.
#
BAND_CHECK_ENABLE = True

# If None: try to infer from seedname and use "<seedname>_band.dat" (and a few common fallbacks).
# Otherwise set explicitly, e.g. "wannier90_band.dat" or "wannier90_band (4).dat".
BAND_DAT_FILE: Optional[str] = None

# Band index in the band.dat file (1-based). If None, use BAND_N.
BAND_CHECK_BAND_INDEX: Optional[int] = None

# Target k-distance (float) where to compute numerical curvature.
# Options:
#   "auto" : choose the internal breakpoint (detected) closest to the middle of the k-path (often Γ).
#   float  : e.g. 1.2604673
#   list   : e.g. [1.2604673, 0.4756798]
BAND_CHECK_KD_TARGET: Union[str, float, List[float]] = "auto"

# Breakpoint detection tolerance (only used for printing candidate high-sym points and for "auto").
BAND_BREAK_REL_TOL = 1e-3
BAND_BREAK_ABS_TOL = 1e-8

# Mismatch warning thresholds between analytic C_total and numerical curvature (best side).
BAND_MATCH_WARN_ABS = 2.0     # eV·Å^2
BAND_MATCH_WARN_REL = 0.35    # relative (|ΔC|/|C|), ignored if |C| is too small

EXPORT_BAND_CURVATURE_CHECK = True
BAND_CURVATURE_CHECK_CSV = "band_curvature_check.csv"




# --- NEW v12: HR numerical derivative/curvature self-check (from hr.dat itself) ---
# This check is *independent* of band.dat and is the most reliable way to validate that the
# analytic (perturbative) curvature formula and the D1/D2 matrices are implemented consistently.
#
# What it does:
#   - Builds H(k0±h) from the same HR file along your chosen direction u_hat
#   - Tracks the target band by maximum overlap with the k0 eigenvector (optional)
#   - Computes numerical derivatives:
#         v_fd  ≈ dE/dk_u
#         C_fd3 ≈ d²E/dk_u²  (3-point central difference)
#         C_fd5 ≈ d²E/dk_u²  (5-point central difference, more accurate)
#     and also one-sided curvature C_onesided(+) from (k0,k0+h,k0+2h) to compare with band.dat.
#   - Compares D1, D2 matrices against finite-difference derivatives of H(k) (sanity check).
#
# Why it helps:
#   - If analytic curvature disagrees with band.dat, this check tells you whether the mismatch
#     comes from (A) the analytic/derivative implementation, or (B) how band.dat curvature is extracted.
#
HR_NUM_CHECK_ENABLE = True

# Base finite-difference step h in Å^-1 along u_hat.
# Options:
#   "band" : if band-check ran and found the target KD, use min(h_left,h_right) from band.dat (typ. ~0.0047 Å^-1)
#   float  : explicit, e.g. 0.001
HR_NUM_CHECK_H: Union[str, float] = "band"
HR_NUM_CHECK_H_FALLBACK = 1e-3  # Å^-1 if "band" is unavailable

# Evaluate multiple step sizes to test convergence (h, h/2, h/4, ...)
HR_NUM_CHECK_H_MULTS = [1.0, 0.5, 0.25]

# Use 5-point second derivative (recommended)
HR_NUM_CHECK_USE_5POINT = True

# Track the band by overlap with the k0 eigenvector (recommended near avoided crossings)
HR_NUM_CHECK_TRACK_BY_OVERLAP = True

EXPORT_HR_NUM_CHECK = True
HR_NUM_CHECK_CSV = "hr_numerical_check.csv"


# =============================================================================
# Step A/B/C extensions ("旋钮" + P0 物理映射)
#
# 你上传的框架文档里，核心是：
#   Step A 先算灵敏度  S_j = ∂C/∂λ_j
#   Step B 用“单个 hopping 缩放旋钮”做 toy 实验（验证方向、非线性等）
#   Step C 用 P0(两份 HR) 或 P2(Harrison+SK) 把真实物理调控 → Δλ_j，再用链式法则预测 ΔC。
#
# 下面参数把这三步写成自动化：
# =============================================================================

# -----------------
# Step A: 旋钮定义
# -----------------
# 旋钮粒度选择：
#   "group"    : λ_g 作用于某个谐波组 g 的所有 H(R) (R 属于该组)
#   "group_ij" : λ_{g,ij} 作用于某个谐波组 g 内、固定 Wannier 轨道对(i,j)的所有 R
#   "Rij"      : λ_{R,ij} 只作用于单个实空间平移 R 的某个 Wannier 轨道对(i,j)
#               （并自动与 (-R,j,i) 做 Hermitian 合并，保证缩放后仍近似 Hermitian）
KNOB_MODE = "Rij"

# 只统计 |q·R| <= KNOB_MAX_GROUP 的 hopping（q 来自 KLINE_START/END）
KNOB_MAX_GROUP = 5

# 若 |H_{ij}(R)| 很小，λ 拟合会非常不稳定；这里直接跳过（并且灵敏度也通常很小）
KNOB_MIN_ABS_T0 = 1e-8

# 导出完整旋钮灵敏度表（可能较大，但在 num_wann~几十、nrpts~几百通常可接受）
EXPORT_KNOB_SENS = True
KNOB_SENS_CSV = "knob_sensitivity.csv"


# --------------------------------------------
# Step B: toy 旋钮实验（指定单个/少数 hopping 缩放）
# --------------------------------------------
KNOB_TUNE_ENABLE = False

# KNOB_TUNE_LIST 只在 KNOB_TUNE_ENABLE=True 时生效。
# 语法：[( (R1,R2,R3), i, j, lambda ), ...]
#   - R 用整数平移（hr.dat 里的 R）
#   - i,j 用 1-based Wannier 轨道编号（与你 log 的 band_n 类似）
#   - lambda 为缩放系数（实数）。脚本会同时缩放 (-R,j,i) 以保持 Hermitian
# 示例：
# KNOB_TUNE_LIST = [
#     ((1, 1, 0),  3,  7, 0.90),
#     ((1, 1, 0),  7,  3, 0.90),  # 不必再写这一条，脚本会自动处理 Hermitian 对
# ]
KNOB_TUNE_LIST: List[Tuple[Tuple[int, int, int], int, int, float]] = []


# ----------------------------------------------------------------
# Step C: P0 真实物理映射（两份 HR：参考/受扰） + ΔC 预测/验证
# ----------------------------------------------------------------
P0_ENABLE = False

# 受扰体系（例如 2% 应变）的 HR/POSCAR/WIN/WOUT/BAND 文件。
# 只要 HR_FILE_P0 给了，其余可以留 None：
#   - lattice 会按 LATTICE_SOURCE_P0 读取（默认沿用 LATTICE_SOURCE）
#   - band.dat 用于“真实曲率”数值校验（可选）
HR_FILE_P0: Optional[str] = None
POSCAR_FILE_P0: Optional[str] = None
WIN_FILE_P0: Optional[str] = None
WOUT_FILE_P0: Optional[str] = None
BAND_DAT_FILE_P0: Optional[str] = None

# P0 比较时用哪个 k 点：
#   "same_k" : 在同一个 k_frac(used) 处比较 C（更贴合链式法则的“固定 k 点”预测）
#   "each_k" : 各自用 min|v| 找到本征 k* 再比较 C（更贴近“带边有效质量”，但包含 k* 漂移）
P0_COMPARE_MODE = "same_k"

# 是否把“晶格/方向变化导致的几何项”单独算出来并加到预测里：
#   ΔC_geom = C(H0, lattice_p0) - C(H0, lattice_ref)
# 这能把纯 hopping 改变与“坐标/方向”的贡献分开。
P0_INCLUDE_GEOMETRY = True

# 在 Step C/P0 模式下，能带序号可能因轻微扰动发生交换。
# 如果两份 HR 处在同一 Wannier gauge（投影/窗口一致且 disentangle 稳定），
# 可以用 |<u_ref|u_p0>| 最大原则追踪同一条能带，从而更稳定比较 ΔC。
P0_BAND_TRACK_BY_OVERLAP = True

# 导出 P0 旋钮映射表：每个 knob 的 λ_fit、Δλ、S、以及预测 ΔC 贡献
EXPORT_P0_KNOB_MAP = True
P0_KNOB_MAP_CSV = "knob_p0_mapping.csv"

# 导出一个汇总：预测 ΔC vs 真实 ΔC
EXPORT_P0_SUMMARY = True
P0_SUMMARY_CSV = "p0_deltaC_summary.csv"

# =============================================================================

# ---- P1 (on-site additive) module: onsite-energy shifts δε_i ----
# Enable this to model knobs like (approx.) electric-field induced onsite shifts,
# deformation potentials, alloying chemistry shifts, etc.
P1_ENABLE                 = False
# CSV format (recommended): columns include either 'i' (1-based) or 'label', plus 'delta_e' (eV).
# Example row: 17,Ga3_s,0.05
P1_ONSITE_CSV             = "P1_onsite_delta.csv"  # set to None to disable CSV reading
P1_FD_DELTA_E             = 1.0e-3   # eV finite-difference step to compute dC/d(ε_i)
P1_APPLY_AND_REEVAL       = True     # compute nonlinear C after applying all δε_i together
P1_EXPORT_SENS_CSV        = "p1_onsite_sensitivity.csv"
P1_EXPORT_SUMMARY_CSV     = "p1_summary.csv"

# ---- P2 (Harrison + Slater–Koster) module: auto-generate λ(R,i,j) from geometry ----
# This maps a structural knob (e.g., strain) -> hopping scalings using
#   (i) Harrison bond-length power law, and optionally
#   (ii) Slater–Koster direction-cosine factors for s/p orbitals.
P2_ENABLE                 = False
# Deformed structure POSCAR (required for P2). If None or missing, P2 is skipped.
P2_POSCAR_DEF             = "P2/POSCAR"
# Optional: deformed wannier90.wout to read deformed Wannier centers. If None, centers are mapped by lattice strain.
P2_WOUT_DEF               = None
P2_USE_SK                 = True     # include SK angle factors (s/p only). If False, use length scaling only.
P2_PP_PI_OVER_SIGMA       = -0.25    # η = V_ppπ / V_ppσ (only used for p–p diagonal terms)
P2_EXPONENT_DEFAULT       = 2.0      # Harrison exponent for s/p interactions (default d^{-2})
P2_EXPONENT_DD            = 5.0      # Harrison exponent for d–d (fallback when label contains 'd')
P2_MIN_ABS_SK             = 1.0e-6   # if |f_old| < this, skip SK ratio and use length scaling only
P2_APPLY_AND_REEVAL       = True     # compute nonlinear C after applying all λ together
P2_EXPORT_LAMBDA_CSV      = "p2_lambda_map.csv"
P2_EXPORT_KNOB_CSV        = "knob_sensitivity_with_p2.csv"

# End USER PARAMETERS


# =============================================================================

# ---------------------------------------------------------------------
# Legacy parameter names (IDE-friendly)
# ---------------------------------------------------------------------
# 说明：
#   下面这些变量名来自你之前的脚本/日志/笔记，为了避免 PyCharm 报“未解析的引用”，
#   这里统一做成“别名”映射到上面已经定义的标准参数。
#   建议你只改上面的标准参数；除非你明确知道这些别名在做什么，否则不要改这里。
#
#   这些别名不会改变物理含义，只是把旧名字指向同一份配置。
P0_KSCAN_TABLE_CSV   = KSCAN_TABLE_CSV          # P0 用的 k 扫描表输出
KNOB_LAMBDA_EPS      = KNOB_MIN_ABS_T0          # |t0| 过小时不做 lambda 拟合/避免发散
KNOB_EXPORT_FULL     = EXPORT_KNOB_SENS         # 是否导出完整 knob 灵敏度表
KNOB_SENSITIVITY_CSV = KNOB_SENS_CSV            # knob 灵敏度表文件名
KNOB_P0_MAPPING_CSV  = P0_KNOB_MAP_CSV          # P0 映射（t1 vs t0）输出
KNOB_SENS_ENABLE     = EXPORT_KNOB_SENS         # 是否启用 knob 灵敏度模块（与 EXPORT_KNOB_SENS 同步）
KNOB_TOPN_PRINT      = 20                       # 控制 knob 灵敏度打印前 N 项
TOPN                 = TOPN_BANDS               # interband top-N 打印（旧名）
POSCAR               = POSCAR_FILE              # 旧名：POSCAR 路径
POSCAR_P0            = POSCAR_FILE_P0           # 旧名：P0 POSCAR 路径
NUM_WANN             = 0                        # 旧名：num_wann（运行时会自动覆盖为 HR 里的真实值）

# Internal / compatibility aliases (to avoid NameError & IDE unresolved refs)
# -----------------------------------------------------------------------------
# Some blocks historically used older variable names. We keep aliases so that
# both naming styles work without editing the main logic.
# =============================================================================
AUTO_K0_MINABS_V_NPTS   = KSCAN_NUM_POINTS
AUTO_K0_REFINE_ROOT     = KSCAN_REFINE_ROOT
AUTO_K0_OVERLAP_TRACK   = KSCAN_TRACK_BY_OVERLAP

P0_KPOINT_MODE          = P0_COMPARE_MODE
P0_KSCAN_CSV            = P0_KSCAN_TABLE_CSV

KNOB_GROUP_MAX              = KNOB_MAX_GROUP
KNOB_MIN_ABS_H0             = KNOB_MIN_ABS_T0
KNOB_DOTR_EPS               = 1e-12
KNOB_ABS_H0_FOR_LAMBDA_EPS  = KNOB_LAMBDA_EPS
KNOB_EXPORT_FULL_TABLE      = KNOB_EXPORT_FULL
KNOB_SENS_CSV               = KNOB_SENSITIVITY_CSV
KNOB_P0_CSV                 = KNOB_P0_MAPPING_CSV

# Legacy aggregate toggle (kept for older branches)
KNOB_ENABLE = bool(KNOB_SENS_ENABLE or KNOB_TUNE_ENABLE or P0_ENABLE)
# =============================================================================


def _normalize_runtime_params() -> None:
    """Synchronize legacy aliases with canonical runtime parameters.

    Canonical knobs (recommended to edit):
      TOPN_BANDS, KNOB_MAX_GROUP, KSCAN_*, P0_COMPARE_MODE,
      ENABLE_P0_PREDICTION / ENABLE_P2_PREDICTION.
    """
    global TOPN_BANDS, TOPN
    global KNOB_MAX_GROUP, KNOB_GROUP_MAX
    global P0_COMPARE_MODE, P0_KPOINT_MODE
    global KSCAN_NUM_POINTS, AUTO_K0_MINABS_V_NPTS
    global KSCAN_REFINE_ROOT, AUTO_K0_REFINE_ROOT
    global KSCAN_TRACK_BY_OVERLAP, AUTO_K0_OVERLAP_TRACK
    global ENABLE_P0_PREDICTION, ENABLE_P2_PREDICTION, DO_P0_PRED, DO_P2_PRED
    global KNOB_SENS_ENABLE, EXPORT_KNOB_SENS, KNOB_ENABLE

    TOPN_BANDS = int(TOPN_BANDS)
    TOPN = int(TOPN_BANDS)

    KNOB_MAX_GROUP = int(KNOB_MAX_GROUP)
    KNOB_GROUP_MAX = int(KNOB_MAX_GROUP)

    P0_KPOINT_MODE = str(P0_COMPARE_MODE)
    AUTO_K0_MINABS_V_NPTS = int(KSCAN_NUM_POINTS)
    AUTO_K0_REFINE_ROOT = bool(KSCAN_REFINE_ROOT)
    AUTO_K0_OVERLAP_TRACK = bool(KSCAN_TRACK_BY_OVERLAP)

    ENABLE_P0_PREDICTION = bool(ENABLE_P0_PREDICTION)
    ENABLE_P2_PREDICTION = bool(ENABLE_P2_PREDICTION)
    DO_P0_PRED = ENABLE_P0_PREDICTION
    DO_P2_PRED = ENABLE_P2_PREDICTION

    EXPORT_KNOB_SENS = bool(KNOB_SENS_ENABLE)
    KNOB_ENABLE = bool(KNOB_SENS_ENABLE or KNOB_TUNE_ENABLE or P0_ENABLE)


HBAR2_OVER_ME = 7.61996424  # eV·Å²

T_R = Tuple[int, int, int]


@dataclass(frozen=True)
class Lattice:
    A: np.ndarray  # (3,3) direct vectors in Å as ROWS
    B: np.ndarray  # (3,3) reciprocal vectors in Å^-1 as ROWS, B @ A.T = 2π I


def read_poscar_lattice(poscar_path: Union[str, Path]) -> Lattice:
    p = Path(poscar_path)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 5:
        raise ValueError("POSCAR too short.")
    scale = float(lines[1].split()[0])
    A_rows = []
    for i in range(2, 5):
        vals = [float(x) for x in lines[i].split()[:3]]
        A_rows.append(vals)
    A = np.array(A_rows, dtype=float) * scale
    B = 2.0 * math.pi * np.linalg.inv(A.T)  # row convention
    return Lattice(A=A, B=B)



def read_win_unit_cell_cart(win_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Parse 'begin unit_cell_cart ... end unit_cell_cart' from a wannier90.win file.
    Returns A as (3,3) direct lattice vectors in Å as ROWS, or None if not found/parse failed.
    """
    p = Path(win_path)
    if not p.exists():
        return None
    ls = p.read_text(encoding='utf-8', errors='ignore').splitlines()
    ib = ie = None
    for i, line in enumerate(ls):
        if line.strip().lower() == 'begin unit_cell_cart':
            ib = i + 1
        if ib is not None and line.strip().lower() == 'end unit_cell_cart':
            ie = i
            break
    if ib is None or ie is None or ie - ib < 3:
        return None
    rows = []
    for j in range(ib, min(ie, ib + 3)):
        parts = ls[j].split()
        if len(parts) < 3:
            return None
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    A = np.array(rows, dtype=float)
    if A.shape != (3, 3):
        return None
    return A


def lattice_from_A_rows(A_rows: np.ndarray) -> Lattice:
    """Build Lattice(A,B) with row convention from A rows in Å."""
    A = np.array(A_rows, dtype=float).reshape(3, 3)
    B = 2.0 * math.pi * np.linalg.inv(A.T)
    return Lattice(A=A, B=B)


def get_lattice_from_inputs(
    poscar_path: Union[str, Path],
    win_path: Optional[Union[str, Path]],
    lattice_source: str = 'auto',
) -> Tuple[Lattice, str]:
    """
    Choose lattice vectors used for dotR (thus curvature units).
    Returns (Lattice, source_used).
    """
    lattice_source = str(lattice_source).strip().lower()
    wp = Path(win_path) if win_path is not None else None
    pp = Path(poscar_path)

    A_win = None
    if wp is not None and wp.exists():
        A_win = read_win_unit_cell_cart(wp)

    # Auto: prefer WIN if available; also warn if WIN and POSCAR disagree significantly.
    if lattice_source in ('auto', 'win') and A_win is not None:
        lat_win = lattice_from_A_rows(A_win)
        if lattice_source == 'auto':
            # Compare with POSCAR if possible
            try:
                lat_pos = read_poscar_lattice(pp)
                # compare vector lengths
                lw = np.linalg.norm(lat_win.A, axis=1)
                lp = np.linalg.norm(lat_pos.A, axis=1)
                rel = np.max(np.abs(lw - lp) / np.maximum(lw, 1e-12))
                if rel > 1e-3:
                    print("[WARN] POSCAR lattice and WIN unit_cell_cart differ. Curvature depends on this!")
                    print(f"       |a| POSCAR: {lp.tolist()} Å")
                    print(f"       |a| WIN   : {lw.tolist()} Å")
                    print("       Using WIN lattice for curvature (LATTICE_SOURCE='auto').")
            except Exception:
                pass
        return lat_win, 'win'

    # POSCAR fallback / forced
    lat_pos = read_poscar_lattice(pp)
    return lat_pos, 'poscar'
def read_wannier90_hr_dat(hr_path: Union[str, Path]) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse Wannier90 seedname_hr.dat.

    Returns:
      num_wann (int)
      nrpts (int)
      R_list (nrpts,3) int
      degeneracy (nrpts,) int
      H_R (nrpts,num_wann,num_wann) complex
    """
    hr_path = Path(hr_path)
    lines = hr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError("hr.dat too short.")
    num_wann = int(lines[1].split()[0])
    nrpts = int(lines[2].split()[0])

    # degeneracies
    deg = []
    idx = 3
    while len(deg) < nrpts:
        parts = lines[idx].split()
        deg.extend(int(x) for x in parts)
        idx += 1
        if idx >= len(lines):
            raise ValueError("EOF while reading degeneracy list.")
    degeneracy = np.array(deg[:nrpts], dtype=np.int64)

    # matrix lines
    expected = nrpts * num_wann * num_wann
    remain = lines[idx:]
    if len(remain) < expected:
        raise ValueError(f"Not enough matrix lines: expect {expected}, got {len(remain)}")

    R_list = np.zeros((nrpts, 3), dtype=np.int64)
    H_R = np.zeros((nrpts, num_wann, num_wann), dtype=np.complex128)

    ptr = 0
    for r in range(nrpts):
        first = remain[ptr].split()
        R1, R2, R3 = int(first[0]), int(first[1]), int(first[2])
        R_list[r] = [R1, R2, R3]
        for _ in range(num_wann * num_wann):
            parts = remain[ptr].split()
            if len(parts) < 7:
                raise ValueError("Bad hr.dat line format.")
            RR = (int(parts[0]), int(parts[1]), int(parts[2]))
            if RR != (R1, R2, R3):
                raise ValueError("Unexpected R ordering in hr.dat block.")
            i = int(parts[3]) - 1
            j = int(parts[4]) - 1
            re = float(parts[5])
            im = float(parts[6])
            H_R[r, i, j] = re + 1j * im
            ptr += 1

    return num_wann, nrpts, R_list, degeneracy, H_R



def read_wannier90_hr(hr_path: str, num_wann_expected: Optional[int] = None):
    """Legacy wrapper kept for backward-compatibility.

    Parameters
    ----------
    hr_path : str
        Path to wannier90_hr.dat
    num_wann_expected : int | None
        If provided and >0, will sanity-check the num_wann inside the file.

    Returns
    -------
    R_list : (nrpts,3) int ndarray
    degeneracy : (nrpts,) int ndarray
    H_R : (nrpts, num_wann, num_wann) complex ndarray
    """
    num_wann, nrpts, R_list, degeneracy, H_R = read_wannier90_hr_dat(hr_path)
    if num_wann_expected is not None and num_wann_expected > 0 and num_wann_expected != num_wann:
        raise ValueError(f"num_wann mismatch: expected {num_wann_expected}, got {num_wann} in {hr_path}")
    return R_list, degeneracy, H_R
def _unit(vec: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < eps:
        raise ValueError(f"Zero vector cannot be normalized: {vec}")
    return vec / n


def build_direction_unit(
    lattice: Lattice,
    dir_mode: str,
    dir_cart: Optional[Sequence[float]] = None,
    kline_start: Optional[Sequence[float]] = None,
    kline_end: Optional[Sequence[float]] = None,
    # Backward/typo-compatible aliases (older driver versions used these names).
    k_line_start: Optional[Sequence[float]] = None,
    k_line_end: Optional[Sequence[float]] = None,
    dir_cart_in: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, str]:
    # Accept legacy keyword 'dir_cart_in' if provided.
    if dir_cart is None and dir_cart_in is not None:
        dir_cart = dir_cart_in

    # Accept legacy keywords 'k_line_start'/'k_line_end' if provided.
    if kline_start is None and k_line_start is not None:
        kline_start = k_line_start
    if kline_end is None and k_line_end is not None:
        kline_end = k_line_end

    if dir_mode == "cart":
        if dir_cart is None:
            raise ValueError("DIR_MODE='cart' requires dir_cart")
        u = np.array(dir_cart, dtype=float)
        return _unit(u), "cart"
    if dir_mode == "from_kline":
        if kline_start is None or kline_end is None:
            raise ValueError("DIR_MODE='from_kline' requires kline_start/kline_end")
        dk = np.array(kline_end, dtype=float) - np.array(kline_start, dtype=float)
        u = dk @ lattice.B
        return _unit(u), "from_kline"
    raise ValueError("DIR_MODE must be 'cart' or 'from_kline'")


def write_csv(path: Union[str, Path], fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})



# ----------------------------------------------------------------------------
# Simple equivalence post-processing for orbpair_by_R_top.csv
# ----------------------------------------------------------------------------
def _strip_atom_index(label: str) -> str:
    """Ga1_px -> Ga_px, As4_pz -> As_pz, otherwise unchanged."""
    m = re.match(r"([A-Za-z]+)\d+_(.+)", str(label))
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return str(label)


def _canonicalize_hermitian_orbpair_row(
    row: Dict[str, object],
    use_dotR: bool = True,
    dotR_eps: float = 1e-10,
) -> Dict[str, object]:
    """
    Canonicalize a row (R,i,j) so that Hermitian-equivalent duplicates map to the same form.

    Equivalence used:
      (R, i, j, label_i, label_j)  ≡  (-R, j, i, label_j, label_i)

    Canonical choice:
      - If |dotR_Ang| > dotR_eps and use_dotR=True: enforce dotR_Ang >= 0 (forward direction).
      - Else: fall back to lexicographic min between (R,i,j) and (-R,j,i).

    Notes:
      - For the conjugate partner, H_im changes sign (complex conjugation).
      - We always store dotR_Ang as a non-negative value in the canonical row.
    """
    R1 = int(row.get("R1", 0))
    R2 = int(row.get("R2", 0))
    R3 = int(row.get("R3", 0))
    i = int(row.get("i", 0))
    j = int(row.get("j", 0))
    li = str(row.get("label_i", ""))
    lj = str(row.get("label_j", ""))
    dotR = float(row.get("dotR_Ang", 0.0))
    H_im = float(row.get("H_im", 0.0))

    flip = False
    if use_dotR and abs(dotR) > dotR_eps:
        flip = dotR < 0.0
    else:
        t1 = (R1, R2, R3, i, j)
        t2 = (-R1, -R2, -R3, j, i)
        flip = t2 < t1

    out = dict(row)
    if flip:
        R1, R2, R3 = -R1, -R2, -R3
        i, j = j, i
        li, lj = lj, li
        dotR = -dotR
        H_im = -H_im  # complex conjugation for canonical orientation

    out["R1"] = int(R1)
    out["R2"] = int(R2)
    out["R3"] = int(R3)
    out["i"] = int(i)
    out["j"] = int(j)
    out["label_i"] = li
    out["label_j"] = lj
    out["dotR_Ang"] = float(abs(dotR))
    out["H_im"] = float(H_im)
    return out


def merge_orbpair_by_R_top_rows(
    rows_topR: List[Dict[str, object]],
    use_dotR: bool = True,
    dotR_eps: float = 1e-10,
    shortlabel: bool = False,
) -> List[Dict[str, object]]:
    """
    Merge Hermitian-equivalent duplicates in orbpair_by_R_top rows.

    Output columns (full mode):
      R1,R2,R3,group,rank_merged,i,j,label_i,label_j,
      n_terms,contrib_sum,contrib_mean,abs_contrib_sum,
      H_re,H_im,absH,weight2_Re,weight2_Im,dotR_Ang,deg,min_rank,max_rank

    If shortlabel=True, the key merges across atom indices using label_i_short/label_j_short,
    and the output keeps those short labels instead of i/j indices.
    """
    agg: Dict[Tuple[object, ...], Dict[str, object]] = {}

    for row in rows_topR:
        r = _canonicalize_hermitian_orbpair_row(row, use_dotR=use_dotR, dotR_eps=dotR_eps)

        R1 = int(r["R1"]); R2 = int(r["R2"]); R3 = int(r["R3"])
        group = int(r.get("group", 0))
        rank0 = int(r.get("rank", 0))
        i = int(r.get("i", 0)); j = int(r.get("j", 0))
        li = str(r.get("label_i", "")); lj = str(r.get("label_j", ""))
        contrib = float(r.get("contrib_intra_ij", 0.0))
        abs_contrib = float(r.get("abs_contrib", abs(contrib)))

        if shortlabel:
            li_k = _strip_atom_index(li)
            lj_k = _strip_atom_index(lj)
            key = (R1, R2, R3, li_k, lj_k)
        else:
            key = (R1, R2, R3, i, j)

        if key not in agg:
            base = {
                "R1": R1, "R2": R2, "R3": R3,
                "group": group,
                "i": i, "j": j,
                "label_i": li, "label_j": lj,
                "n_terms": 0,
                "contrib_sum": 0.0,
                "abs_contrib_sum": 0.0,
                "H_re": float(r.get("H_re", 0.0)),
                "H_im": float(r.get("H_im", 0.0)),
                "absH": float(r.get("absH", 0.0)),
                "weight2_Re": float(r.get("weight2_Re", 0.0)),
                "weight2_Im": float(r.get("weight2_Im", 0.0)),
                "dotR_Ang": float(r.get("dotR_Ang", 0.0)),
                "deg": int(r.get("deg", 1)),
                "min_rank": rank0,
                "max_rank": rank0,
            }
            if shortlabel:
                base["label_i_short"] = _strip_atom_index(li)
                base["label_j_short"] = _strip_atom_index(lj)
            agg[key] = base

        a = agg[key]
        a["n_terms"] = int(a["n_terms"]) + 1
        a["contrib_sum"] = float(a["contrib_sum"]) + contrib
        a["abs_contrib_sum"] = float(a["abs_contrib_sum"]) + abs_contrib
        a["min_rank"] = min(int(a["min_rank"]), rank0)
        a["max_rank"] = max(int(a["max_rank"]), rank0)

    # finalize: mean + rank within each R
    merged: List[Dict[str, object]] = []
    # group by R for ranking
    buckets: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for a in agg.values():
        n = int(a["n_terms"])
        a["contrib_mean"] = float(a["contrib_sum"]) / float(n) if n > 0 else 0.0
        a["pair_complete"] = 1 if n == 2 else 0
        Rk = (int(a["R1"]), int(a["R2"]), int(a["R3"]))
        buckets.setdefault(Rk, []).append(a)

    for Rk, arr in buckets.items():
        arr_sorted = sorted(arr, key=lambda x: -abs(float(x.get("contrib_sum", 0.0))))
        for rrk, a in enumerate(arr_sorted, start=1):
            out = dict(a)
            out["rank_merged"] = rrk
            merged.append(out)

    # stable ordering: by R then rank_merged
    merged.sort(key=lambda x: (int(x["R1"]), int(x["R2"]), int(x["R3"]), int(x["rank_merged"])))
    return merged



# =============================================================================
# Knob utilities (single hopping scaling + P0 mapping)
# =============================================================================

def best_real_scale(x0: np.ndarray, x1: np.ndarray, eps: float = 1e-14) -> Tuple[float, float]:
    """Return the best *real* scalar λ that fits x1 ≈ λ x0 in least squares.

    Works for complex vectors.

    Returns:
        lam: best-fit real scalar
        rel_resid: ||x1 - lam x0||^2 / ||x1||^2   (0 is perfect, 1 is bad)
    """
    x0 = np.asarray(x0, dtype=np.complex128).ravel()
    x1 = np.asarray(x1, dtype=np.complex128).ravel()
    den = float(np.vdot(x0, x0).real)  # Σ |x0|^2
    if den < eps:
        return 1.0, 0.0
    num = np.vdot(x0, x1)  # Σ conj(x0) x1 (complex)
    lam = float(num.real / den)
    resid = np.vdot(x1 - lam * x0, x1 - lam * x0).real
    tot = np.vdot(x1, x1).real
    rel_resid = float(resid / tot) if tot > eps else 0.0
    return lam, rel_resid


def canonical_herm_key_Rij(
    R: Tuple[int, int, int],
    i: int,
    j: int,
    dotR: float,
    eps_dotR: float = 1e-10,
) -> Tuple[Tuple[int, int, int], int, int]:
    """Canonical key for a Hermitian pair:

        (R, i, j)  <->  (-R, j, i)

    We prefer the member with dotR>0 along the chosen direction (u-hat).
    If dotR≈0, fall back to lexicographic ordering.
    """
    R1, R2, R3 = R
    Rp = (-R1, -R2, -R3)
    ip, jp = j, i
    if dotR > eps_dotR:
        return (R1, R2, R3), i, j
    if dotR < -eps_dotR:
        return Rp, ip, jp
    # dotR ~ 0
    if (R1, R2, R3, i, j) <= (Rp[0], Rp[1], Rp[2], ip, jp):
        return (R1, R2, R3), i, j
    return Rp, ip, jp


def apply_single_hop_scaling_inplace(
    H_R: np.ndarray,
    R_to_idx: Dict[Tuple[int, int, int], int],
    R: Tuple[int, int, int],
    i0: int,
    j0: int,
    lam: float,
    keep_hermitian: bool = True,
) -> None:
    """Scale a single hopping H_{i0,j0}(R) by a real factor lam.

    If keep_hermitian=True (recommended), we also scale the Hermitian partner:
        H_{j0,i0}(-R)
    """
    idx = R_to_idx[R]
    H_R[idx, i0, j0] *= lam
    if keep_hermitian:
        Rp = (-R[0], -R[1], -R[2])
        idxp = R_to_idx[Rp]
        H_R[idxp, j0, i0] *= lam




def build_knob_table_Rij(*args, **kwargs):
    """Backward-compatible alias. Prefer compute_knob_table_Rij()."""
    return compute_knob_table_Rij(*args, **kwargs)


def apply_knob_scalings_Rij(
    H_R_in: np.ndarray,
    R_list: Sequence[Tuple[int, int, int]],
    scalings: Sequence[Dict[str, object]],
    *,
    one_based_orb: bool = True,
) -> np.ndarray:
    """
    Apply a list of (R,i,j,lambda) scalings to HR on a COPY and return it.

    Each item in `scalings` can be:
      - dict with keys: R (iterable len=3) or (R1,R2,R3), i, j, lambda
      - or dict with keys: R1,R2,R3,i,j,lam/lambda/scale

    Orbit indices i,j are 1-based by default (set one_based_orb=False to use 0-based).
    The scaling λ can be real/complex.
    """
    H_R = np.array(H_R_in, copy=True)
    R_to_idx = {tuple(map(int, R)): idx for idx, R in enumerate(R_list)}

    for item in scalings:
        if not isinstance(item, dict):
            raise TypeError(f"Each scaling must be a dict, got {type(item)}")

        if "R" in item:
            R = tuple(int(x) for x in item["R"])  # type: ignore[arg-type]
        else:
            R = (int(item.get("R1")), int(item.get("R2")), int(item.get("R3")))

        i = int(item.get("i"))
        j = int(item.get("j"))
        if one_based_orb:
            i -= 1
            j -= 1

        lam = item.get("lambda", item.get("lam", item.get("scale", 1.0)))

        # allow complex scalings: either complex, or (lam_re, lam_im)
        if isinstance(lam, (list, tuple)) and len(lam) == 2:
            lam_val = complex(float(lam[0]), float(lam[1]))
        else:
            lam_val = complex(lam)

        apply_single_hop_scaling_inplace(H_R, R_to_idx, R, i, j, lam_val)

    return H_R
def compute_knob_table_Rij(
    *,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    H_R: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    dotR: np.ndarray,
    group_labels: List[int],
    q_vec_int: Tuple[int, int, int],
    band_vec_n: np.ndarray,
    S_vec: np.ndarray,
    labels: List[str],
    group_max: int = 5,
    min_abs_t0: float = 1e-8,
    eps_dotR: float = 1e-10,
    H_R_p0: Optional[np.ndarray] = None,
    R_to_idx_p0: Optional[Dict[Tuple[int, int, int], int]] = None,
) -> List[Dict[str, object]]:
    """Build a Hermitian-merged per-(R,i,j) knob table.

    Each row corresponds to the physical knob that scales both:
        H_{ij}(R) and H_{ji}(-R)

    Sensitivities are linear-response at λ=1 (eigenvectors/denominators fixed).
    If H_R_p0 is provided, also compute best-fit λ (P0) and predicted ΔC.
    """
    nw = H_R.shape[1]
    assert H_R.shape[2] == nw
    R_to_idx = {tuple(map(int, R_list[r])): r for r in range(R_list.shape[0])}

    vn0 = np.asarray(band_vec_n, dtype=np.complex128).reshape(-1)
    S = np.asarray(S_vec, dtype=np.complex128).reshape(-1)

    rows: List[Dict[str, object]] = []
    for idxR in range(R_list.shape[0]):
        R = tuple(map(int, R_list[idxR]))
        g = int(group_labels[idxR])
        if g > group_max:
            continue

        # If dotR=0 for the chosen u, then w1=w2=0 and this R cannot contribute to
        # v/curvature along u (its knob sensitivity will be exactly zero). Skip to
        # keep tables compact and rankings meaningful.
        if abs(w1[idxR]) < 1e-18 and abs(w2[idxR]) < 1e-18:
            continue

        # Precompute per-element sensitivity matrices for this R
        M2 = w2[idxR] * H_R[idxR]
        intra_mat = np.real(np.conj(vn0)[:, None] * M2 * vn0[None, :])
        M1 = w1[idxR] * H_R[idxR]
        inter_mat = 4.0 * np.real(np.conj(vn0)[:, None] * M1 * S[None, :])

        for i in range(nw):
            for j in range(nw):
                # canonical representative check
                keyR, keyi, keyj = canonical_herm_key_Rij(R, i, j, float(dotR[idxR]), eps_dotR=eps_dotR)
                if (keyR != R) or (keyi != i) or (keyj != j):
                    continue

                Rp = (-R[0], -R[1], -R[2])
                if Rp not in R_to_idx:
                    continue
                idxRp = R_to_idx[Rp]

                t0_f = complex(H_R[idxR, i, j])
                t0_p = complex(H_R[idxRp, j, i])
                if max(abs(t0_f), abs(t0_p)) < min_abs_t0:
                    continue

                # partner sensitivity (may be in same R if R=0)
                intra_f = float(intra_mat[i, j])
                inter_f = float(inter_mat[i, j])
                if idxRp == idxR and j == i:
                    intra_pair = intra_f
                    inter_pair = inter_f
                else:
                    # compute matrices for partner R on demand
                    M2p = w2[idxRp] * H_R[idxRp]
                    intra_mat_p = np.real(np.conj(vn0)[:, None] * M2p * vn0[None, :])
                    M1p = w1[idxRp] * H_R[idxRp]
                    inter_mat_p = 4.0 * np.real(np.conj(vn0)[:, None] * M1p * S[None, :])
                    intra_pair = intra_f + float(intra_mat_p[j, i])
                    inter_pair = inter_f + float(inter_mat_p[j, i])

                total_pair = intra_pair + inter_pair

                row: Dict[str, object] = {
                    "R1": R[0],
                    "R2": R[1],
                    "R3": R[2],
                    "group_abs_n": g,
                    "q1": q_vec_int[0],
                    "q2": q_vec_int[1],
                    "q3": q_vec_int[2],
                    "i": i + 1,
                    "j": j + 1,
                    "label_i": labels[i] if i < len(labels) else f"w{i+1}",
                    "label_j": labels[j] if j < len(labels) else f"w{j+1}",
                    "dotR_Ang": float(dotR[idxR]),
                    "t0_f_re": float(np.real(t0_f)),
                    "t0_f_im": float(np.imag(t0_f)),
                    "t0_p_re": float(np.real(t0_p)),
                    "t0_p_im": float(np.imag(t0_p)),
                    "abs_t0_f": float(abs(t0_f)),
                    "abs_t0_p": float(abs(t0_p)),
                    "S_intra": intra_pair,
                    "S_inter": inter_pair,
                    "S_total": total_pair,
                }

                # Optional P0 mapping
                if H_R_p0 is not None and R_to_idx_p0 is not None:
                    if R in R_to_idx_p0 and Rp in R_to_idx_p0:
                        idxR1 = R_to_idx_p0[R]
                        idxRp1 = R_to_idx_p0[Rp]
                        t1_f = complex(H_R_p0[idxR1, i, j])
                        t1_p = complex(H_R_p0[idxRp1, j, i])
                        lam_fit, rel_res = best_real_scale(np.array([t0_f, t0_p]), np.array([t1_f, t1_p]))
                        dlam = lam_fit - 1.0
                        row.update({
                            "t1_f_re": float(np.real(t1_f)),
                            "t1_f_im": float(np.imag(t1_f)),
                            "t1_p_re": float(np.real(t1_p)),
                            "t1_p_im": float(np.imag(t1_p)),
                            "abs_t1_f": float(abs(t1_f)),
                            "abs_t1_p": float(abs(t1_p)),
                            "lambda_fit": float(lam_fit),
                            "delta_lambda": float(dlam),
                            "fit_rel_resid": float(rel_res),
                            "pred_dC_intra": float(intra_pair * dlam),
                            "pred_dC_inter": float(inter_pair * dlam),
                            "pred_dC_total": float(total_pair * dlam),
                        })
                    else:
                        row.update({
                            "lambda_fit": np.nan,
                            "delta_lambda": np.nan,
                            "fit_rel_resid": np.nan,
                            "pred_dC_total": np.nan,
                        })

                rows.append(row)

    return rows




# =============================================================================
# P1 / P2 built-in modules (Step-B knobs) + Step-C validation
# =============================================================================

def _idx_map_from_R_list(R_list: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    """Map integer lattice vectors R -> idx in the hr arrays."""
    mp: Dict[Tuple[int, int, int], int] = {}
    for idx, R in enumerate(R_list):
        mp[(int(R[0]), int(R[1]), int(R[2]))] = idx
    return mp


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def apply_lambda_map_to_hr(
    H_R: np.ndarray,
    R_list: np.ndarray,
    lambda_map: Dict[Tuple[int, int, int, int, int], float],
) -> np.ndarray:
    """Return a *new* H_R with element-wise scalings applied on Hermitian pairs.

    lambda_map keys are (R1,R2,R3,i,j) with i,j 1-based and (R,i,j) already
    Hermitian-canonicalized. We apply the same λ to (R,i,j) and (-R,j,i).
    """
    H_new = H_R.copy()
    ridx = _idx_map_from_R_list(R_list)
    for (R1, R2, R3, i1, j1), lam in lambda_map.items():
        if abs(lam - 1.0) < 1e-16:
            continue
        Rt = (int(R1), int(R2), int(R3))
        Rm = (-int(R1), -int(R2), -int(R3))
        idx_R = ridx.get(Rt, None)
        idx_Rm = ridx.get(Rm, None)
        if idx_R is None or idx_Rm is None:
            continue
        i = int(i1) - 1
        j = int(j1) - 1
        H_new[idx_R, i, j] *= lam
        H_new[idx_Rm, j, i] *= lam
    return H_new


def apply_onsite_deltas_to_hr(
    H_R: np.ndarray,
    R_list: np.ndarray,
    onsite_deltas: Dict[int, float],
) -> np.ndarray:
    """Apply onsite shifts δε_i (additive) on the R=(0,0,0) block diagonal."""
    H_new = H_R.copy()
    ridx = _idx_map_from_R_list(R_list)
    idx_R0 = ridx.get((0, 0, 0), None)
    if idx_R0 is None:
        raise RuntimeError("R=(0,0,0) not found in HR R_list; cannot apply onsite deltas")
    for i1, de in onsite_deltas.items():
        i = int(i1) - 1
        H_new[idx_R0, i, i] += float(de)
    return H_new


def compute_total_curvature_from_hr(
    H_R: np.ndarray,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    lattice: Lattice,
    k_frac: np.ndarray,
    u_hat_cart: np.ndarray,
    band_n_1based: int,
) -> Tuple[float, float, float]:
    """Compute (C_intra, C_inter, C_total) for band_n at given k and direction."""
    _phase, w0, w1, w2, _dotR = build_weights_and_dotR(
        R_list=R_list,
        degeneracy=degeneracy,
        lattice=lattice,
        k_frac=k_frac,
        u_hat_cart=u_hat_cart,
    )
    Hk, D1, D2 = build_H_D1_D2(w0, w1, w2, H_R)
    res = curvature_and_interband_table(
        Hk=Hk,
        D1=D1,
        D2=D2,
        band_n_1based=band_n_1based,
        band_m_1based=None,
    )
    C_intra = float(res["C_intra"])
    C_inter = float(res["C_inter"])
    C_total = float(res["C_total"])
    return float(C_intra), float(C_inter), float(C_total)


def read_wannier_centers_from_wout(
    wout_file: str,
    num_wann: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Parse Wannier centers (Angstrom) from wannier90.wout.

    Returns array shape (num_wann,3) in Cartesian Angstrom, or None if not found.
    """
    if wout_file is None:
        return None
    if not os.path.exists(wout_file):
        return None
    centers: List[List[float]] = []
    pat = re.compile(r"WF centre and spread\s+\d+\s+\(\s*([\-\+0-9Ee\.]+)\s*,\s*([\-\+0-9Ee\.]+)\s*,\s*([\-\+0-9Ee\.]+)\s*\)")
    try:
        with open(wout_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "WF centre and spread" not in line:
                    continue
                m = pat.search(line)
                if not m:
                    continue
                centers.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        if num_wann is not None:
            if len(centers) < num_wann:
                return None
            centers = centers[:num_wann]
        if len(centers) == 0:
            return None
        return np.array(centers, dtype=float)
    except Exception:
        return None


def _orb_kind_axis_from_label(label: str) -> Tuple[str, Optional[str]]:
    """Very lightweight parser: 'Ga1_px' -> ('p','x'); 'Ga3_s' -> ('s',None).

    Unknown labels return ('?', None).
    """
    if label is None:
        return "?", None
    s = str(label).strip()
    if s == "":
        return "?", None
    # take the last token after '_'
    tok = s.split("_")[-1].lower()
    if tok in ("s", "s0", "s1"):
        return "s", None
    if tok in ("px", "py", "pz"):
        return "p", tok[-1]
    if tok.startswith("d"):
        return "d", tok  # keep subkind
    return "?", None


def _harrison_exponent_from_labels(
    label_i: str,
    label_j: str,
    n_default: float,
    n_dd: float,
) -> float:
    ki, _ai = _orb_kind_axis_from_label(label_i)
    kj, _aj = _orb_kind_axis_from_label(label_j)
    if ki == "d" and kj == "d":
        return float(n_dd)
    # for now: use default for everything else (s/p and mixed)
    return float(n_default)


def _sk_factor_sp(
    axis_p: str,
    l: float,
    m: float,
    n: float,
) -> float:
    if axis_p == "x":
        return float(l)
    if axis_p == "y":
        return float(m)
    if axis_p == "z":
        return float(n)
    return 0.0


def _sk_factor_pp(
    axis_a: str,
    axis_b: str,
    l: float,
    m: float,
    n: float,
    eta_pp: float,
) -> float:
    # p–p Slater-Koster (using η = Vπ/Vσ).
    if axis_a == axis_b:
        # diagonal (px-px, py-py, pz-pz)
        if axis_a == "x":
            c2 = l * l
        elif axis_a == "y":
            c2 = m * m
        else:
            c2 = n * n
        return float(c2 + (1.0 - c2) * eta_pp)
    # off-diagonal (px-py etc): proportional to l*m*(1-η), ratio cancels (1-η)
    if {axis_a, axis_b} == {"x", "y"}:
        return float(l * m)
    if {axis_a, axis_b} == {"x", "z"}:
        return float(l * n)
    if {axis_a, axis_b} == {"y", "z"}:
        return float(m * n)
    return 0.0


def _sk_factor(
    label_i: str,
    label_j: str,
    l: float,
    m: float,
    n: float,
    eta_pp: float,
) -> Optional[float]:
    ki, ai = _orb_kind_axis_from_label(label_i)
    kj, aj = _orb_kind_axis_from_label(label_j)
    if ki == "s" and kj == "s":
        return 1.0
    if ki == "s" and kj == "p" and aj is not None:
        return _sk_factor_sp(aj, l, m, n)
    if ki == "p" and ai is not None and kj == "s":
        return _sk_factor_sp(ai, l, m, n)
    if ki == "p" and kj == "p" and ai is not None and aj is not None:
        return _sk_factor_pp(ai, aj, l, m, n, eta_pp)
    return None


def _parse_onsite_csv(
    csv_path: Optional[str],
    labels: List[str],
) -> Dict[int, float]:
    """Read onsite shifts from CSV. Returns dict of {i_1based: delta_e}."""
    if csv_path is None:
        return {}
    if not os.path.exists(csv_path):
        return {}
    out: Dict[int, float] = {}
    # build label -> index (1-based) map
    lab2i = {str(lab): idx + 1 for idx, lab in enumerate(labels)}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        # allow also headerless CSV: try sniff
        if reader.fieldnames is None:
            return out
        for row in reader:
            if row is None:
                continue
            # accept keys: i/idx, label, delta_e/de
            i_str = (row.get("i") or row.get("idx") or row.get("index") or "").strip()
            lab = (row.get("label") or row.get("wannier") or "").strip()
            de_str = (row.get("delta_e") or row.get("de") or row.get("delta") or "").strip()
            if de_str == "":
                continue
            try:
                de = float(de_str)
            except Exception:
                continue
            i1: Optional[int] = None
            if i_str != "":
                try:
                    i1 = int(float(i_str))
                except Exception:
                    i1 = None
            if i1 is None and lab != "":
                i1 = lab2i.get(lab, None)
            if i1 is None:
                continue
            out[int(i1)] = out.get(int(i1), 0.0) + float(de)
    return out


def run_P1_onsite_additive(
    H_R: np.ndarray,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    lattice: Lattice,
    k_frac: np.ndarray,
    u_hat_cart: np.ndarray,
    band_n_1based: int,
    labels: List[str],
    onsite_csv: Optional[str],
    fd_delta_e: float = 1.0e-3,
    apply_and_reeval: bool = True,
    export_sens_csv: Optional[str] = None,
    export_summary_csv: Optional[str] = None,
) -> None:
    onsite_deltas = _parse_onsite_csv(onsite_csv, labels)
    if len(onsite_deltas) == 0:
        print("[P1] No onsite deltas found. Skip P1.")
        return

    C0_intra, C0_inter, C0 = compute_total_curvature_from_hr(
        H_R, R_list, degeneracy, lattice, k_frac, u_hat_cart, band_n_1based
    )
    print("\n=== [P1] On-site additive module ===")
    print(f"[P1] baseline C_total = {C0:+.6e} eV·Å^2")
    print(f"[P1] onsite knobs (count) = {len(onsite_deltas)}; FD delta = {fd_delta_e:g} eV")

    # finite-difference sensitivities dC/dε_i
    ridx = _idx_map_from_R_list(R_list)
    idx_R0 = ridx.get((0, 0, 0), None)
    if idx_R0 is None:
        print("[P1][WARN] R=(0,0,0) not found; cannot run P1.")
        return

    rows: List[Dict[str, object]] = []
    pred_dC = 0.0
    for i1, de in sorted(onsite_deltas.items(), key=lambda kv: kv[0]):
        i0 = int(i1) - 1
        lab = labels[i0] if 0 <= i0 < len(labels) else f"w{i1}"
        H_fd = H_R.copy()
        H_fd[idx_R0, i0, i0] += float(fd_delta_e)
        _ci, _ce, C_fd = compute_total_curvature_from_hr(
            H_fd, R_list, degeneracy, lattice, k_frac, u_hat_cart, band_n_1based
        )
        dC_dEi = (C_fd - C0) / float(fd_delta_e)
        dC_pred_i = float(dC_dEi) * float(de)
        pred_dC += dC_pred_i
        rows.append({
            "i": int(i1),
            "label": lab,
            "delta_e_eV": float(de),
            "dC_dEi_eVA2_per_eV": float(dC_dEi),
            "dC_pred_eVA2": float(dC_pred_i),
        })

    print(f"[P1] linear prediction ΔC = {pred_dC:+.6e} eV·Å^2")

    C_new = None
    if apply_and_reeval:
        H_new = apply_onsite_deltas_to_hr(H_R, R_list, onsite_deltas)
        _ci2, _ce2, C_new = compute_total_curvature_from_hr(
            H_new, R_list, degeneracy, lattice, k_frac, u_hat_cart, band_n_1based
        )
        dC_true = float(C_new - C0)
        print(f"[P1] nonlinear re-eval C_total = {C_new:+.6e} eV·Å^2  (ΔC_true={dC_true:+.6e})")
        print(f"[P1] linear vs nonlinear: ΔC_pred - ΔC_true = {pred_dC - dC_true:+.6e}")

    if export_sens_csv:
        write_csv(
            export_sens_csv,
            fieldnames=["i", "label", "delta_e_eV", "dC_dEi_eVA2_per_eV", "dC_pred_eVA2"],
            rows=rows,
        )
        print(f"[P1][OUT] Wrote onsite sensitivity table to: {export_sens_csv}")

    if export_summary_csv:
        summ = [{
            "C0": float(C0),
            "C0_intra": float(C0_intra),
            "C0_inter": float(C0_inter),
            "dC_pred": float(pred_dC),
            "C_new": (float(C_new) if C_new is not None else ""),
            "dC_true": (float(C_new - C0) if C_new is not None else ""),
            "fd_delta_e": float(fd_delta_e),
            "num_knobs": int(len(onsite_deltas)),
        }]
        write_csv(
            export_summary_csv,
            fieldnames=["C0", "C0_intra", "C0_inter", "dC_pred", "C_new", "dC_true", "fd_delta_e", "num_knobs"],
            rows=summ,
        )
        print(f"[P1][OUT] Wrote P1 summary to: {export_summary_csv}")


def run_P2_harrison_sk(
    H_R: np.ndarray,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    lattice_ref: Lattice,
    poscar_def: str,
    wout_ref: Optional[str],
    wout_def: Optional[str],
    k_frac: np.ndarray,
    u_hat_cart: np.ndarray,
    band_n_1based: int,
    labels: List[str],
    knob_rows: List[Dict[str, object]],
    use_sk: bool = True,
    eta_pp: float = -0.25,
    n_default: float = 2.0,
    n_dd: float = 5.0,
    min_abs_sk: float = 1.0e-6,
    apply_and_reeval: bool = True,
    export_lambda_csv: Optional[str] = None,
    export_knob_csv: Optional[str] = None,
) -> None:
    if poscar_def is None or (not os.path.exists(poscar_def)):
        print("[P2] POSCAR_DEF missing; skip P2.")
        return
    centers_ref = read_wannier_centers_from_wout(wout_ref, num_wann=len(labels))
    if centers_ref is None:
        print("[P2][WARN] Cannot read Wannier centers from wout_ref. P2 needs centers; skip.")
        return
    centers_def = read_wannier_centers_from_wout(wout_def, num_wann=len(labels)) if wout_def else None
    lattice_def = read_poscar_lattice(poscar_def)
    A0 = np.array(lattice_ref.A, dtype=float)
    A1 = np.array(lattice_def.A, dtype=float)
    try:
        F = np.linalg.inv(A0) @ A1  # row-vector convention: d1 = d0 @ F
    except Exception as e:
        print(f"[P2][WARN] Failed to build deformation gradient from POSCAR: {e}; skip.")
        return

    # baseline curvature
    _c0i, _c0e, C0 = compute_total_curvature_from_hr(
        H_R, R_list, degeneracy, lattice_ref, k_frac, u_hat_cart, band_n_1based
    )

    print("\n=== [P2] Harrison + Slater–Koster module ===")
    print(f"[P2] POSCAR_def = {poscar_def}")
    print(f"[P2] use_SK={use_sk}, eta_pp={eta_pp}, n_default={n_default}, n_dd={n_dd}")
    if centers_def is not None:
        print(f"[P2] deformed centers: using {wout_def}")
    else:
        print("[P2] deformed centers: None (use lattice strain mapping)")

    # build λ map for each Hermitian-merged knob row
    lambda_map: Dict[Tuple[int, int, int, int, int], float] = {}
    detail_rows: List[Dict[str, object]] = []
    pred_dC = 0.0

    for row in knob_rows:
        R1 = int(row["R1"]); R2 = int(row["R2"]); R3 = int(row["R3"])
        i1 = int(row["i"]); j1 = int(row["j"])
        lab_i = labels[i1 - 1] if 1 <= i1 <= len(labels) else f"w{i1}"
        lab_j = labels[j1 - 1] if 1 <= j1 <= len(labels) else f"w{j1}"
        Ri = np.array([R1, R2, R3], dtype=float)

        # bond vectors
        t0 = Ri @ A0
        d0 = (centers_ref[j1 - 1] + t0) - centers_ref[i1 - 1]
        r0 = _safe_norm(d0)
        if centers_def is not None:
            t1 = Ri @ A1
            d1 = (centers_def[j1 - 1] + t1) - centers_def[i1 - 1]
        else:
            d1 = d0 @ F
        r1 = _safe_norm(d1)

        if r0 < 1e-12 or r1 < 1e-12:
            lam_len = 1.0
            lam_ang = 1.0
            lam = 1.0
        else:
            n_exp = _harrison_exponent_from_labels(lab_i, lab_j, n_default, n_dd)
            lam_len = (r0 / r1) ** float(n_exp)

            lam_ang = 1.0
            if use_sk:
                l0, m0, n0 = (d0 / r0).tolist()
                l1, m1, n1 = (d1 / r1).tolist()
                f0 = _sk_factor(lab_i, lab_j, l0, m0, n0, eta_pp=eta_pp)
                f1 = _sk_factor(lab_i, lab_j, l1, m1, n1, eta_pp=eta_pp)
                if (f0 is None) or (f1 is None) or (abs(float(f0)) < float(min_abs_sk)):
                    lam_ang = 1.0
                else:
                    lam_ang = float(f1) / float(f0)

            lam = float(lam_len) * float(lam_ang)

        key = (R1, R2, R3, i1, j1)
        lambda_map[key] = float(lam)

        dlam = float(lam) - 1.0
        S_tot = float(row.get("S_total", 0.0))
        dC_pred = S_tot * dlam
        pred_dC += dC_pred

        detail_rows.append({
            "R1": R1, "R2": R2, "R3": R3, "i": i1, "j": j1,
            "label_i": lab_i, "label_j": lab_j,
            "r0_A": float(r0), "r1_A": float(r1),
            "lambda_len": float(lam_len), "lambda_ang": float(lam_ang), "lambda": float(lam),
            "S_total": float(S_tot), "dC_pred": float(dC_pred),
        })

    print(f"[P2] linear prediction ΔC = {pred_dC:+.6e} eV·Å^2")

    # Attach to knob rows and export
    if export_knob_csv:
        knob_out: List[Dict[str, object]] = []
        for row in knob_rows:
            R1 = int(row["R1"]); R2 = int(row["R2"]); R3 = int(row["R3"])
            i1 = int(row["i"]); j1 = int(row["j"])
            lam = float(lambda_map.get((R1, R2, R3, i1, j1), 1.0))
            dlam = lam - 1.0
            S_tot = float(row.get("S_total", 0.0))
            row2 = dict(row)
            row2.update({
                "lambda_P2": lam,
                "dlam_P2": dlam,
                "dC_pred_P2": float(S_tot * dlam),
            })
            knob_out.append(row2)
        write_csv(
            export_knob_csv,
            fieldnames=list(knob_out[0].keys()) if knob_out else list(knob_rows[0].keys()) + ["lambda_P2", "dlam_P2", "dC_pred_P2"],
            rows=knob_out,
        )
        print(f"[P2][OUT] Wrote knob table (+P2 λ) to: {export_knob_csv}")

    if export_lambda_csv:
        write_csv(
            export_lambda_csv,
            fieldnames=["R1", "R2", "R3", "i", "j", "label_i", "label_j", "r0_A", "r1_A", "lambda_len", "lambda_ang", "lambda", "S_total", "dC_pred"],
            rows=detail_rows,
        )
        print(f"[P2][OUT] Wrote P2 λ table to: {export_lambda_csv}")

    # Step-C: nonlinear re-evaluation (toy but self-consistent within TB)
    if apply_and_reeval:
        H_new = apply_lambda_map_to_hr(H_R, R_list, lambda_map)
        _ci2, _ce2, C_new = compute_total_curvature_from_hr(
            H_new, R_list, degeneracy, lattice_ref, k_frac, u_hat_cart, band_n_1based
        )
        dC_true = float(C_new - C0)
        print(f"[P2] nonlinear re-eval C_total = {C_new:+.6e} eV·Å^2  (ΔC_true={dC_true:+.6e})")
        print(f"[P2] linear vs nonlinear: ΔC_pred - ΔC_true = {pred_dC - dC_true:+.6e}")

# ----------------------------------------------------------------------------
# Numerical curvature check from Wannier90 band file (seedname_band.dat)
# ----------------------------------------------------------------------------
def _read_wannier90_band_dat(band_path: Union[str, Path]) -> List[np.ndarray]:
    """
    Read Wannier90 seedname_band.dat (or similar) file.

    Format: blocks separated by blank lines; each line typically:
        k_distance   energy(eV)
    Some builds may include extra columns; we only take the first two numbers.

    Returns:
        bands: list of arrays of shape (Nk,2)
    """
    p = Path(band_path)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    bands: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            if cur:
                bands.append(cur)
                cur = []
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except Exception:
            continue
        cur.append((x, y))
    if cur:
        bands.append(cur)

    out = [np.array(b, dtype=float) for b in bands if len(b) >= 3]
    return out


def _detect_kpath_breakpoints(kdist: np.ndarray, rel_tol: float = 1e-3, abs_tol: float = 1e-8) -> List[int]:
    """
    Detect indices of likely high-symmetry points / segment junctions along the band-plot path
    using changes in k-distance step.

    Returns list of indices including [0] and [N-1].
    """
    kdist = np.asarray(kdist, dtype=float).reshape(-1)
    n = kdist.size
    if n < 3:
        return [0, n - 1] if n > 0 else []

    d = np.diff(kdist)
    # If there are repeated points (d≈0), treat as a breakpoint indicator as well.
    break_idxs = {0, n - 1}

    for i in range(1, d.size):
        a = float(d[i - 1])
        b = float(d[i])
        if abs(a) < abs_tol and abs(b) < abs_tol:
            continue
        # breakpoint if step changes a lot OR one of them is ~0 (duplicate at junction)
        if abs(b - a) > max(abs_tol, rel_tol * max(abs(a), abs(b))) or (abs(a) < abs_tol) or (abs(b) < abs_tol):
            # change occurs at point i (between steps into and out of i)
            break_idxs.add(i)

    return sorted(break_idxs)


def _quad_second_derivative(x: np.ndarray, y: np.ndarray) -> float:
    """Fit y(x) with a quadratic and return d²y/dx² (constant) = 2a."""
    coeff = np.polyfit(np.asarray(x, float), np.asarray(y, float), 2)
    a = float(coeff[0])
    return 2.0 * a


def _quad_first_derivative_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Fit quadratic and return dy/dx at x0."""
    a, b, _c = np.polyfit(np.asarray(x, float), np.asarray(y, float), 2)
    return float(2.0 * a * x0 + b)


def _resolve_band_dat_path(hr_path: Path) -> Optional[Path]:
    """
    Resolve band file path.

    If BAND_DAT_FILE is provided, use it.
    Else try common candidates based on inferred seedname.
    """
    if BAND_DAT_FILE is not None:
        p = Path(str(BAND_DAT_FILE))
        if p.exists():
            return p
        # also try relative to HR directory
        p2 = hr_path.parent / p.name
        if p2.exists():
            return p2
        return None

    seed = _infer_seedname_from_hr(hr_path)
    candidates = [
        f"{seed}_band.dat",
        f"{seed}.band.dat",
        "wannier90_band.dat",
        "wannier90_band.dat",  # duplicate ok
    ]
    for c in candidates:
        p = hr_path.parent / c
        if p.exists():
            return p
    return None


def run_band_curvature_check(
    hr_path: Path,
    band_n_1based: int,
    C_analytic: float,
    u_mode: str,
) -> List[Dict[str, object]]:
    """
    Run numerical curvature check from band file, print results, and return rows for CSV export.

    Args:
        hr_path: path to HR file (for seedname inference / relative search)
        band_n_1based: which band in band file
        C_analytic: analytic total curvature (eV·Å^2)
        u_mode: "cart" or "from_kline" (for printing suggestion)

    Returns:
        list of rows (dict) for export
    """
    band_path = _resolve_band_dat_path(hr_path)
    if band_path is None:
        print("[WARN] BAND_CHECK_ENABLE=True but band.dat file not found. Set BAND_DAT_FILE explicitly to enable check.")
        print("")
        return []

    bands = _read_wannier90_band_dat(band_path)
    if not bands:
        print(f"[WARN] Could not read any band blocks from: {band_path}")
        print("")
        return []

    bidx = int(band_n_1based) - 1
    if not (0 <= bidx < len(bands)):
        print(f"[WARN] band index {band_n_1based} out of range for {band_path} (has {len(bands)} bands).")
        print("")
        return []

    arr = bands[bidx]
    kdist = arr[:, 0]
    E = arr[:, 1]
    N = kdist.size

    break_idxs = _detect_kpath_breakpoints(kdist, rel_tol=float(BAND_BREAK_REL_TOL), abs_tol=float(BAND_BREAK_ABS_TOL))

    # Print breakpoints for user reference
    print("=== Band-file numerical curvature self-check ===")
    print(f"band.dat  : {band_path}")
    print(f"band_idx  : {band_n_1based} (1-based)")
    print(f"analytic C_total (hr) : {C_analytic:+.10e} eV·Å^2")
    print("")
    print("Detected k-path breakpoints (index, kdist):")
    for ii in break_idxs:
        print(f"  {ii:>5d}  {kdist[ii]:.8f}")
    print("")

    # Choose targets
    targets: List[int] = []
    if isinstance(BAND_CHECK_KD_TARGET, str) and BAND_CHECK_KD_TARGET.lower() == "auto":
        internal = [ii for ii in break_idxs if 2 <= ii <= N - 3]  # need both sides
        if not internal:
            # fallback: pick middle index with margins
            ii = max(2, min(N - 3, N // 2))
            targets = [ii]
        else:
            mid = 0.5 * (kdist[0] + kdist[-1])
            ii = min(internal, key=lambda j: abs(float(kdist[j]) - float(mid)))
            targets = [ii]
    elif isinstance(BAND_CHECK_KD_TARGET, (float, int)):
        t = float(BAND_CHECK_KD_TARGET)
        ii = int(np.argmin(np.abs(kdist - t)))
        targets = [ii]
    elif isinstance(BAND_CHECK_KD_TARGET, list):
        for t0 in BAND_CHECK_KD_TARGET:
            t = float(t0)
            ii = int(np.argmin(np.abs(kdist - t)))
            targets.append(ii)
        # unique keep order
        seen = set()
        targets2 = []
        for ii in targets:
            if ii not in seen:
                targets2.append(ii); seen.add(ii)
        targets = targets2
    else:
        # fallback
        ii = max(2, min(N - 3, N // 2))
        targets = [ii]

    rows_out: List[Dict[str, object]] = []

    for ii in targets:
        kd = float(kdist[ii])
        En = float(E[ii])

        # left side (ii-2,ii-1,ii)
        C_left = None
        v_left = None
        h_left = None
        if ii >= 2:
            x = kdist[ii - 2: ii + 1]
            y = E[ii - 2: ii + 1]
            C_left = float(_quad_second_derivative(x, y))
            v_left = float(_quad_first_derivative_at(x, y, kd))
            h_left = float(kdist[ii] - kdist[ii - 1])

        # right side (ii,ii+1,ii+2)
        C_right = None
        v_right = None
        h_right = None
        if ii <= N - 3:
            x = kdist[ii: ii + 3]
            y = E[ii: ii + 3]
            C_right = float(_quad_second_derivative(x, y))
            v_right = float(_quad_first_derivative_at(x, y, kd))
            h_right = float(kdist[ii + 1] - kdist[ii])

        # Compare to analytic
        diffs = []
        if C_left is not None:
            diffs.append(("left", abs(C_left - C_analytic), C_left))
        if C_right is not None:
            diffs.append(("right", abs(C_right - C_analytic), C_right))

        if diffs:
            best_side, best_diff, best_C = sorted(diffs, key=lambda t: t[1])[0]
        else:
            best_side, best_diff, best_C = ("none", float("nan"), float("nan"))

        # Warnings
        warn = False
        rel = None
        if math.isfinite(best_diff):
            if abs(C_analytic) > 1e-8:
                rel = best_diff / abs(C_analytic)
                if best_diff > float(BAND_MATCH_WARN_ABS) and rel > float(BAND_MATCH_WARN_REL):
                    warn = True
            else:
                if best_diff > float(BAND_MATCH_WARN_ABS):
                    warn = True

        # Print
        print(f"Target index {ii}  kdist={kd:.8f}  E={En:+.8f} eV")
        if C_left is not None:
            print(f"  left : h={h_left:.6e}  v={v_left:+.6e} eV·Å   C={C_left:+.6e} eV·Å^2")
        else:
            print("  left : (not enough points)")
        if C_right is not None:
            print(f"  right: h={h_right:.6e}  v={v_right:+.6e} eV·Å   C={C_right:+.6e} eV·Å^2")
        else:
            print("  right: (not enough points)")

        if diffs:
            side_note = "toward decreasing kdist" if best_side == "left" else "toward increasing kdist"
            print(f"  best match vs analytic: {best_side} ({side_note}), |ΔC|={best_diff:.3e}")
            if warn:
                                print("  [WARN] Analytic curvature does NOT match band-file curvature well.")
                                if u_mode != 'from_kline':
                                    print("         Most common cause: direction mismatch.")
                                    print("         Suggestion: set DIR_MODE='from_kline' and choose KLINE_START/KLINE_END to match the band segment.")
                                    print(f"         Current DIR_MODE='{u_mode}'.")
                                else:
                                    print("         DIR_MODE is already 'from_kline'. Remaining common causes:")
                                    print("           (1) Lattice mismatch for derivatives: POSCAR vs <seedname>.win unit_cell_cart (units/strain).")
                                    print("               Energies use only phase exp(i2π k·R) and can match even if lattice is wrong, but curvature will not.")
                                    print("           (2) BAND_DAT_FILE is not generated from the same HR_FILE / seedname.")
                                    print("         Suggestion: set LATTICE_SOURCE='win' (recommended) or ensure POSCAR lattice matches .win exactly.")
        print("")

        rows_out.append({
            "band_dat": str(band_path),
            "band_index": int(band_n_1based),
            "target_idx": int(ii),
            "kdist": kd,
            "E": En,
            "C_left": "" if C_left is None else C_left,
            "C_right": "" if C_right is None else C_right,
            "best_side": best_side,
            "C_best": best_C,
            "C_analytic": C_analytic,
            "abs_diff_best": best_diff,
            "rel_diff_best": "" if rel is None else rel,
            "v_left": "" if v_left is None else v_left,
            "v_right": "" if v_right is None else v_right,
            "h_left": "" if h_left is None else h_left,
            "h_right": "" if h_right is None else h_right,
        })

    if EXPORT_BAND_CURVATURE_CHECK and rows_out:
        write_csv(
            BAND_CURVATURE_CHECK_CSV,
            fieldnames=["band_dat", "band_index", "target_idx", "kdist", "E",
                        "C_left", "C_right", "best_side", "C_best",
                        "C_analytic", "abs_diff_best", "rel_diff_best",
                        "v_left", "v_right", "h_left", "h_right"],
            rows=rows_out,
        )
        print(f"[OUT] Wrote band curvature check table to: {BAND_CURVATURE_CHECK_CSV}")
        print("")

    return rows_out



def _norm_fro(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat, ord="fro"))


def _herm(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.conj().T)


def _build_Hk_only(
    lattice: Lattice,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    H_R: np.ndarray,
    k_frac: Sequence[float],
) -> np.ndarray:
    # H(k) does not depend on u_hat; only the phase does.
    k = np.array(k_frac, dtype=float)
    phase = np.exp(1j * 2.0 * math.pi * (R_list @ k))
    w0 = phase / degeneracy.astype(float)
    Hk = np.tensordot(w0, H_R, axes=(0, 0))
    return _herm(Hk)


def _kfrac_shift_from_h(
    lattice: Lattice,
    u_hat_cart: np.ndarray,
    h: float,
) -> np.ndarray:
    '''
    Convert a small Cartesian reciprocal-space step delta_k_cart = h * u_hat (Å^-1)
    into a fractional k step delta_k_frac such that:
        delta_k_cart = delta_k_frac @ lattice.B
    where lattice.B has reciprocal vectors as ROWS (Å^-1).
    '''
    delta_cart = float(h) * np.asarray(u_hat_cart, float)  # Å^-1
    Binv = np.linalg.inv(lattice.B)
    delta_frac = delta_cart @ Binv
    return np.asarray(delta_frac, float)


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _infer_band_step_from_bandcheck_rows(rows: List[Dict[str, object]]) -> Optional[float]:
    '''
    Given band_curvature_check rows (for a given target kdist), return a reasonable
    finite-difference step h (Å^-1). We take the smallest positive among h_left/h_right.
    '''
    hs: List[float] = []
    for r in rows:
        hl = _safe_float(r.get("h_left", None))
        hr = _safe_float(r.get("h_right", None))
        if hl is not None and hl > 0:
            hs.append(float(hl))
        if hr is not None and hr > 0:
            hs.append(float(hr))
    if not hs:
        return None
    return float(min(hs))


def run_hr_numerical_check(
    lattice: Lattice,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    H_R: np.ndarray,
    k0_frac: Sequence[float],
    u_hat_cart: np.ndarray,
    band_n_1based: int,
    Hk0: np.ndarray,
    D10: np.ndarray,
    D20: np.ndarray,
    evals0: np.ndarray,
    evecs0: np.ndarray,
    C_analytic: float,
    bandcheck_rows: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    '''
    Numerical derivative/curvature check directly from HR (independent of band.dat).
    Returns rows for CSV export.
    '''
    if not HR_NUM_CHECK_ENABLE:
        return []

    # choose base h
    if isinstance(HR_NUM_CHECK_H, str) and HR_NUM_CHECK_H.lower() == "band":
        hb = None
        if bandcheck_rows:
            hb = _infer_band_step_from_bandcheck_rows(bandcheck_rows)
        if hb is None:
            hb = float(HR_NUM_CHECK_H_FALLBACK)
        h_base = float(hb)
    else:
        h_base = float(HR_NUM_CHECK_H)

    mults = list(HR_NUM_CHECK_H_MULTS) if isinstance(HR_NUM_CHECK_H_MULTS, list) else [1.0]

    # analytic band vector at k0
    n0 = band_n_1based - 1
    v0 = evecs0[:, n0]
    v0 = v0 / np.linalg.norm(v0)

    v_analytic = float(np.real(np.vdot(v0, D10 @ v0)))
    C_intra_analytic = float(np.real(np.vdot(v0, D20 @ v0)))

    print("=== HR numerical derivative/curvature self-check (independent of band.dat) ===")
    print(f"band_n (k0 eigenvalue index): {band_n_1based}")
    print(f"analytic v  = <n|dH/dk|n>        : {v_analytic:+.10e} eV·Å")
    print(f"analytic C_intra = <n|d²H/dk²|n> : {C_intra_analytic:+.10e} eV·Å^2")
    print(f"analytic C_total (intra+inter)  : {C_analytic:+.10e} eV·Å^2")
    print(f"h_base = {h_base:.6e} Å^-1, mults={mults}")
    print("")

    rows_out: List[Dict[str, object]] = []

    nD1 = _norm_fro(D10)
    nD2 = _norm_fro(D20)

    for mult in mults:
        h = float(h_base) * float(mult)
        if h <= 0:
            continue

        dk_frac = _kfrac_shift_from_h(lattice, u_hat_cart, h)

        k0 = np.asarray(k0_frac, float)
        kp = k0 + dk_frac
        km = k0 - dk_frac
        kpp = k0 + 2.0 * dk_frac
        kmm = k0 - 2.0 * dk_frac

        # Build H at shifted points
        Hkp = _build_Hk_only(lattice, R_list, degeneracy, H_R, kp)
        Hkm = _build_Hk_only(lattice, R_list, degeneracy, H_R, km)

        evals_p, evecs_p = np.linalg.eigh(Hkp)
        evals_m, evecs_m = np.linalg.eigh(Hkm)

        if HR_NUM_CHECK_TRACK_BY_OVERLAP:
            idx_p, ov_p = band_track_index(evecs0, evecs_p, band_n_1based)
            idx_m, ov_m = band_track_index(evecs0, evecs_m, band_n_1based)
        else:
            idx_p, ov_p = band_n_1based, float("nan")
            idx_m, ov_m = band_n_1based, float("nan")

        Ep = float(evals_p[idx_p - 1].real)
        Em = float(evals_m[idx_m - 1].real)
        E0 = float(evals0[band_n_1based - 1].real)

        v_fd = (Ep - Em) / (2.0 * h)
        C_fd3 = (Ep - 2.0 * E0 + Em) / (h ** 2)

        # one-sided (+) curvature for comparison to band.dat (k0,k0+h,k0+2h)
        Hkpp = _build_Hk_only(lattice, R_list, degeneracy, H_R, kpp)
        evals_pp, evecs_pp = np.linalg.eigh(Hkpp)
        if HR_NUM_CHECK_TRACK_BY_OVERLAP:
            idx_pp, ov_pp = band_track_index(evecs0, evecs_pp, band_n_1based)
        else:
            idx_pp, ov_pp = band_n_1based, float("nan")
        Epp = float(evals_pp[idx_pp - 1].real)
        C_onesided_p = (Epp - 2.0 * Ep + E0) / (h ** 2)

        # one-sided (-) curvature (k0,k0-h,k0-2h)
        Hkmm = _build_Hk_only(lattice, R_list, degeneracy, H_R, kmm)
        evals_mm, evecs_mm = np.linalg.eigh(Hkmm)
        if HR_NUM_CHECK_TRACK_BY_OVERLAP:
            idx_mm, ov_mm = band_track_index(evecs0, evecs_mm, band_n_1based)
        else:
            idx_mm, ov_mm = band_n_1based, float("nan")
        Emm = float(evals_mm[idx_mm - 1].real)
        C_onesided_m = (E0 - 2.0 * Em + Emm) / (h ** 2)

        C_fd5_val: Optional[float] = None
        if HR_NUM_CHECK_USE_5POINT:
            # 5-point centered second derivative:
            C_fd5_val = (-Epp + 16.0 * Ep - 30.0 * E0 + 16.0 * Em - Emm) / (12.0 * (h ** 2))

        # Matrix finite-difference checks (independent of band tracking)
        D1_fd = _herm((Hkp - Hkm) / (2.0 * h))
        D2_fd = _herm((Hkp - 2.0 * Hk0 + Hkm) / (h ** 2))

        rel_err_D1 = _norm_fro(D1_fd - D10) / (nD1 + 1e-18)
        rel_err_D2 = _norm_fro(D2_fd - D20) / (nD2 + 1e-18)

        # Print
        print(f"h = {h:.6e} Å^-1  (mult={mult:g})")
        print(f"  tracked idx (+h,-h,+2h,-2h): {idx_p},{idx_m},{idx_pp},{idx_mm}  overlaps: {ov_p:.4f},{ov_m:.4f},{ov_pp:.4f},{ov_mm:.4f}")
        print(f"  v_fd   (central)           : {v_fd:+.10e} eV·Å   |Δv|={abs(v_fd - v_analytic):.3e}")
        print(f"  C_fd3  (central)           : {C_fd3:+.10e} eV·Å^2 |ΔC|={abs(C_fd3 - C_analytic):.3e}")
        if C_fd5_val is not None:
            print(f"  C_fd5  (5-point)           : {C_fd5_val:+.10e} eV·Å^2 |ΔC|={abs(C_fd5_val - C_analytic):.3e}")
        print(f"  C_onesided(+) (k,k+h,k+2h) : {C_onesided_p:+.10e} eV·Å^2")
        print(f"  C_onesided(-) (k,k-h,k-2h) : {C_onesided_m:+.10e} eV·Å^2")
        print(f"  ||D1_fd - D1||/||D1||      : {rel_err_D1:.3e}")
        print(f"  ||D2_fd - D2||/||D2||      : {rel_err_D2:.3e}")
        print("")

        row = {
            "h": h,
            "mult": float(mult),
            "E0": E0,
            "Ep": Ep,
            "Em": Em,
            "Epp": Epp,
            "Emm": Emm,
            "idx_p": idx_p,
            "idx_m": idx_m,
            "idx_pp": idx_pp,
            "idx_mm": idx_mm,
            "ov_p": ov_p,
            "ov_m": ov_m,
            "ov_pp": ov_pp,
            "ov_mm": ov_mm,
            "v_analytic": v_analytic,
            "v_fd": v_fd,
            "C_analytic": C_analytic,
            "C_fd3": C_fd3,
            "C_fd5": ("" if C_fd5_val is None else C_fd5_val),
            "C_onesided_p": C_onesided_p,
            "C_onesided_m": C_onesided_m,
            "rel_err_D1": rel_err_D1,
            "rel_err_D2": rel_err_D2,
        }
        rows_out.append(row)

    if EXPORT_HR_NUM_CHECK and rows_out:
        write_csv(
            HR_NUM_CHECK_CSV,
            fieldnames=[
                "h", "mult", "E0", "Ep", "Em", "Epp", "Emm",
                "idx_p", "idx_m", "idx_pp", "idx_mm",
                "ov_p", "ov_m", "ov_pp", "ov_mm",
                "v_analytic", "v_fd",
                "C_analytic", "C_fd3", "C_fd5",
                "C_onesided_p", "C_onesided_m",
                "rel_err_D1", "rel_err_D2",
            ],
            rows=rows_out,
        )
        print(f"[OUT] Wrote HR numerical check table to: {HR_NUM_CHECK_CSV}")
        print("")

    return rows_out


def rationalize_delta_k(delta_k: Sequence[float], denom_max: int) -> Tuple[np.ndarray, int, List[Fraction]]:
    fracs = [Fraction(x).limit_denominator(denom_max) for x in delta_k]

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b) if a and b else abs(a or b)

    D = 1
    for f in fracs:
        D = lcm(D, f.denominator)

    q = np.array([int(f.numerator * (D // f.denominator)) for f in fracs], dtype=np.int64)

    g = 0
    for x in q.tolist():
        g = math.gcd(g, abs(int(x)))
    if g > 1:
        q = q // g
        D = D // g

    return q, D, fracs


def harmonic_group_labels(
    R_list: np.ndarray,
    q: np.ndarray,
    max_abs_n: Optional[int] = None,
) -> Tuple[List[Union[int, str]], Dict[Union[int, str], np.ndarray], np.ndarray, np.ndarray]:
    n_raw = (R_list @ q.reshape(3, 1)).reshape(-1).astype(np.int64)
    abs_n = np.abs(n_raw)

    labels: List[Union[int, str]] = []
    if max_abs_n is None:
        labels = [int(x) for x in abs_n.tolist()]
    else:
        big_label = f">{max_abs_n}"
        for x in abs_n.tolist():
            labels.append(int(x) if int(x) <= max_abs_n else big_label)

    group_to_idx: Dict[Union[int, str], List[int]] = {}
    for i, g in enumerate(labels):
        group_to_idx.setdefault(g, []).append(i)

    group_to_indices = {g: np.array(idxs, dtype=np.int64) for g, idxs in group_to_idx.items()}
    return labels, group_to_indices, n_raw, abs_n


def build_weights_and_dotR(
    lattice: Lattice,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    k_frac: Sequence[float],
    u_hat_cart: Optional[np.ndarray] = None,
    # Backward-compatible alias: some call sites used keyword 'u_hat'
    u_hat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      phase (nrpts,) complex = exp(i2π k·R)
      w0    (nrpts,) complex = phase/deg
      w1    (nrpts,) complex = i dotR * w0          for D1
      w2    (nrpts,) complex = -(dotR^2) * w0       for D2
      dotR  (nrpts,) float   in Å
    """
    # Accept alias keyword 'u_hat' if provided.
    if u_hat_cart is None and u_hat is not None:
        u_hat_cart = u_hat
    if u_hat_cart is None:
        raise ValueError("u_hat_cart (or u_hat) is required")

    k = np.array(k_frac, dtype=float)
    phase = np.exp(1j * 2.0 * math.pi * (R_list @ k))
    w0 = phase / degeneracy.astype(float)

    R_abs = R_list @ lattice.A
    dotR = (R_abs @ u_hat_cart).astype(float)

    w1 = 1j * dotR * w0
    w2 = -(dotR ** 2) * w0
    return phase, w0, w1, w2, dotR


def build_H_D1_D2(
    w0: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    H_R: np.ndarray,
    scale_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build H(k), D1, D2 using precomputed weights.
    If scale_override (nrpts,) is given, multiply each R block by that scale (ablation).
    """
    if scale_override is None:
        s = 1.0
        Hk = np.tensordot(w0, H_R, axes=(0, 0))
        D1 = np.tensordot(w1, H_R, axes=(0, 0))
        D2 = np.tensordot(w2, H_R, axes=(0, 0))
    else:
        w0s = w0 * scale_override
        w1s = w1 * scale_override
        w2s = w2 * scale_override
        Hk = np.tensordot(w0s, H_R, axes=(0, 0))
        D1 = np.tensordot(w1s, H_R, axes=(0, 0))
        D2 = np.tensordot(w2s, H_R, axes=(0, 0))

    # Hermitize (numerical)
    Hk = 0.5 * (Hk + Hk.conj().T)
    D1 = 0.5 * (D1 + D1.conj().T)
    D2 = 0.5 * (D2 + D2.conj().T)
    return Hk, D1, D2


def band_track_index(
    evecs_ref: np.ndarray,
    evecs_new: np.ndarray,
    idx_ref_1based: int,
) -> Tuple[int, float]:
    idx_ref = idx_ref_1based - 1
    vref = evecs_ref[:, idx_ref]
    vref = vref / np.linalg.norm(vref)
    overlaps = np.abs(evecs_new.conj().T @ vref)
    j = int(np.argmax(overlaps))
    return j + 1, float(overlaps[j])


def curvature_and_interband_table(
    Hk: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    band_n_1based: int,
    band_m_1based: Optional[int] = None,
) -> Dict[str, object]:
    evals, evecs = np.linalg.eigh(Hk)
    nb = evals.size
    n = band_n_1based - 1
    En = float(evals[n].real)
    vn = evecs[:, n]

    C_intra = float(np.real(np.vdot(vn, D2 @ vn)))

    D1_eig = evecs.conj().T @ D1 @ evecs
    Vn = D1_eig[n, :]
    V2 = np.abs(Vn) ** 2

    dE = En - evals
    contrib = np.zeros(nb, dtype=float)
    for m in range(nb):
        if m == n:
            continue
        denom = float(dE[m].real)
        contrib[m] = float(2.0 * V2[m].real / denom)

    C_inter = float(np.sum(contrib) - contrib[n])
    C_total = C_intra + C_inter

    rows = []
    for m in range(nb):
        if m == n:
            continue
        rows.append({
            "m": m + 1,
            "Em": float(evals[m].real),
            "dE": float((En - evals[m]).real),
            "V2": float(V2[m].real),
            "contrib": float(contrib[m]),
        })

    pair = None
    if band_m_1based is not None and (1 <= band_m_1based <= nb) and band_m_1based != band_n_1based:
        m = band_m_1based - 1
        pair = {
            "m": band_m_1based,
            "Em": float(evals[m].real),
            "dE": float((En - evals[m]).real),
            "V2": float(V2[m].real),
            "contrib": float(contrib[m]),
        }

    return {
        "evals": evals,
        "evecs": evecs,
        "En": En,
        "C_intra": C_intra,
        "C_inter": C_inter,
        "C_total": C_total,
        "inter_table": rows,
        "pair": pair,
    }



def _infer_seedname_from_hr(hr_path: Path) -> str:
    """
    Infer Wannier90 seedname from HR file name.
    Typical: seedname_hr.dat  -> seedname
    """
    name = hr_path.name
    if name.endswith("_hr.dat"):
        return name[:-7]
    if name.endswith(".hr.dat"):
        return name[:-7]
    if name.endswith(".dat") and "_hr" in name:
        return name.split("_hr")[0]
    return hr_path.stem


def _read_poscar_atom_list(poscar_path: Path) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Parse POSCAR element symbols and counts (VASP5-style). Returns:
      atom_symbols (len=natoms) : global atom list, e.g. ["Fe","Fe","O",...]
      atom_elem_ord (len=natoms): ordinal within element, e.g. [1,2,1,2,3,...]
      species (list)
      counts (list)
    If element symbols are not present (old POSCAR), species are set to X1,X2,...
    """
    lines = poscar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 8:
        raise ValueError("POSCAR too short to parse atoms.")

    # After lattice: line indices 0-based: 5 and 6 typically
    tokens5 = lines[5].split()
    def _is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except Exception:
            return False

    if tokens5 and all(_is_int(t) for t in tokens5):
        # Old POSCAR: no species line
        counts = [int(t) for t in tokens5]
        species = [f"X{i+1}" for i in range(len(counts))]
    else:
        species = tokens5
        tokens6 = lines[6].split()
        if not tokens6 or not all(_is_int(t) for t in tokens6):
            raise ValueError("Failed to parse POSCAR counts line.")
        counts = [int(t) for t in tokens6]

    atom_symbols: List[str] = []
    for sym, n in zip(species, counts):
        atom_symbols.extend([sym] * int(n))

    # ordinal within each element
    elem_counter: Dict[str, int] = {}
    atom_elem_ord: List[int] = []
    for sym in atom_symbols:
        elem_counter[sym] = elem_counter.get(sym, 0) + 1
        atom_elem_ord.append(elem_counter[sym])

    return atom_symbols, atom_elem_ord, species, counts


def _canonical_orbital_list(tag: str) -> List[str]:
    """Expand common orbital sets and normalize common synonyms.

    Notes
    -----
    Wannier90 projections often use lists like "s; p" or "s,p". We treat ';' and ',' as list separators.
    """
    t_raw = tag.strip().lower()
    if not t_raw:
        return []
    # keep a copy with spaces removed for token matching
    t = t_raw.replace(" ", "")
    if not t:
        return []

    # split common multi-token lists (commas / semicolons), but do not break l=/m= expressions
    if ("l=" not in t and "m=" not in t) and ("," in t or ";" in t):
        t2 = t.replace(";", ",")
        out: List[str] = []
        for part in t2.split(","):
            part = part.strip()
            if not part:
                continue
            out.extend(_canonical_orbital_list(part))
        return out

    # explicit real harmonics / common names
    # d
    if "dx2-y2" in t or "dx2-y^2" in t or "dx2y2" in t:
        return ["dx2-y2"]
    if "d3z2-r2" in t or "dz2" in t or "dz^2" in t or "d(z2)" in t:
        return ["dz2"]
    if "dxy" in t:
        return ["dxy"]
    if "dxz" in t:
        return ["dxz"]
    if "dyz" in t:
        return ["dyz"]

    # p
    if "p_x" in t or t == "px":
        return ["px"]
    if "p_y" in t or t == "py":
        return ["py"]
    if "p_z" in t or t == "pz":
        return ["pz"]

    # sets
    if t in ("s", "l=0"):
        return ["s"]
    if t in ("p", "l=1"):
        return ["px", "py", "pz"]
    if t in ("d", "l=2"):
        return ["dxy", "dyz", "dz2", "dxz", "dx2-y2"]
    if t in ("f", "l=3"):
        return [f"f{i}" for i in range(1, 8)]

    # l=, m=
    m_l = re.search(r"l\s*=\s*([0-3])", t_raw.lower())
    if m_l:
        l = int(m_l.group(1))
        m_m = re.search(r"m\s*=\s*([-]?\d+)", t_raw.lower())
        if m_m:
            return [f"l{l}_m{m_m.group(1)}"]
        if l == 0:
            return ["s"]
        if l == 1:
            return ["px", "py", "pz"]
        if l == 2:
            return ["dxy", "dyz", "dz2", "dxz", "dx2-y2"]
        if l == 3:
            return [f"f{i}" for i in range(1, 8)]

    # hybrids
    if "sp3" in t:
        return ["sp3_1", "sp3_2", "sp3_3", "sp3_4"]
    if "sp2" in t:
        return ["sp2_1", "sp2_2", "sp2_3"]
    if t == "sp":
        return ["sp_1", "sp_2"]

    # fallback: keep raw token (sanitized later)
    return [t]


def _sanitize_label(s: str, maxlen: int = 40) -> str:
    s2 = s.strip().replace(" ", "")
    s2 = s2.replace("(", "").replace(")", "")
    s2 = s2.replace("[", "").replace("]", "")
    s2 = s2.replace("{", "").replace("}", "")
    s2 = s2.replace(",", "_").replace(":", "_")
    if len(s2) > maxlen:
        s2 = s2[:maxlen]
    return s2


def _labels_from_win(win_path: Path, atom_symbols: List[str], atom_elem_ord: List[int], num_wann: int) -> List[str]:
    """
    Build a best-effort WF label list by expanding 'begin projections' block in seedname.win.

    Important: Wannier90 allows (very common) syntax like:
        Ga : s; p
    meaning *the same* left site (Ga) with multiple orbital sets (s and p).
    Semicolons may also separate multiple full specs on one line:
        Ga:s; As:p

    This function handles both correctly.
    """
    lines = win_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    in_proj = False
    specs: List[Tuple[str, str]] = []

    def split_projection_line(s: str) -> List[Tuple[str, str]]:
        """
        Split one projections line into a list of (left, right_expr) specs.

        Example:
          "Ga : s; p"     -> [("Ga", "s,p")]
          "Ga:s; As:p"    -> [("Ga", "s"), ("As", "p")]
          "p"             -> [("", "p")]
        """
        # Remove inline comments
        s2 = s.split("#", 1)[0].split("!", 1)[0].strip()
        if not s2:
            return []
        # Split into semicolon segments
        segs = [seg.strip() for seg in s2.split(";") if seg.strip()]
        if not segs:
            return []

        out: List[Tuple[str, str]] = []
        cur_left: Optional[str] = None
        cur_orb_tokens: List[str] = []

        def flush():
            nonlocal cur_left, cur_orb_tokens
            if cur_left is None:
                return
            right_expr = ",".join([t.strip() for t in cur_orb_tokens if t.strip()])
            out.append((cur_left, right_expr))
            cur_left = None
            cur_orb_tokens = []

        for seg in segs:
            if ":" in seg:
                flush()
                left, right = [x.strip() for x in seg.split(":", 1)]
                cur_left = left
                cur_orb_tokens = [right.strip()] if right.strip() else []
            else:
                # continuation orbital token
                if cur_left is None:
                    # site-less spec (rare, but valid)
                    cur_left = ""
                    cur_orb_tokens = [seg.strip()]
                else:
                    cur_orb_tokens.append(seg.strip())

        flush()
        return out

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("!"):
            continue
        low = s.lower()
        if low.startswith("begin projections"):
            in_proj = True
            continue
        if low.startswith("end projections"):
            in_proj = False
            continue
        if in_proj:
            specs.extend(split_projection_line(s))

    if not specs:
        return []

    # precompute indices by element
    species_set = set(atom_symbols)
    elem_to_atom_indices: Dict[str, List[int]] = {}
    for gi, sym in enumerate(atom_symbols, start=1):
        elem_to_atom_indices.setdefault(sym, []).append(gi)

    def atom_base(global_idx: int) -> str:
        sym = atom_symbols[global_idx - 1]
        ord_ = atom_elem_ord[global_idx - 1]
        return f"{sym}{ord_}"

    labels: List[str] = []

    for left, right_expr in specs:
        sp_left = (left or "").strip()
        sp_right = (right_expr or "").strip()
        sp_full = f"{sp_left}:{sp_right}" if sp_left else sp_right

        low_full = sp_full.lower().replace(" ", "")
        if not low_full or low_full in ("random", "none"):
            continue

        # Determine sites
        sites: List[str]
        left_clean = sp_left.replace(" ", "")
        if not left_clean:
            sites = [""]
        else:
            lcl = left_clean
            lcl_low = lcl.lower()
            if lcl_low.startswith("c=") or lcl_low.startswith("center=") or lcl_low.startswith("site="):
                sites = [_sanitize_label(lcl)]
            elif lcl.isdigit():
                idx = int(lcl)
                if 1 <= idx <= len(atom_symbols):
                    sites = [atom_base(idx)]
                else:
                    sites = [f"Atom{idx}"]
            elif lcl in species_set:
                sites = [atom_base(i) for i in elem_to_atom_indices.get(lcl, [])]
            else:
                # unknown tag, keep as one "site"
                sites = [_sanitize_label(lcl)]

        orbs = _canonical_orbital_list(sp_right)

        if not orbs:
            for base in sites:
                labels.append(base if base else _sanitize_label(sp_full))
        else:
            for base in sites:
                for orb in orbs:
                    if base:
                        labels.append(f"{base}_{orb}")
                    else:
                        labels.append(_sanitize_label(f"{sp_full}_{orb}"))

    return labels


def _extract_orbital_keyword(line: str) -> Optional[str]:
    """Try to find a recognizable orbital token in a line (for .wout heuristics)."""
    s = line.lower()
    # order matters: match longer tokens first
    candidates = [
        ("dx2-y2", ["dx2-y2", "dx2-y^2", "dx2y2"]),
        ("dz2", ["d3z2-r2", "dz2", "dz^2"]),
        ("dxy", ["dxy"]),
        ("dxz", ["dxz"]),
        ("dyz", ["dyz"]),
        ("px", ["p_x", "px"]),
        ("py", ["p_y", "py"]),
        ("pz", ["p_z", "pz"]),
        ("sp3", ["sp3"]),
        ("sp2", ["sp2"]),
        ("sp", [" sp "]),
        ("p", [" p "]),
        ("d", [" d "]),
        ("s", [" s "]),
    ]
    for canon, keys in candidates:
        for k in keys:
            if k.strip() in ("p", "d", "s") and k not in s:
                continue
            if k in s:
                return canon
    # try l=, m=
    m_l = re.search(r"l\s*=\s*([0-3])", s)
    if m_l:
        l = int(m_l.group(1))
        m_m = re.search(r"m\s*=\s*([-]?\d+)", s)
        if m_m:
            return f"l{l}_m{m_m.group(1)}"
        if l == 0:
            return "s"
        if l == 1:
            return "p"
        if l == 2:
            return "d"
        if l == 3:
            return "f"
    return None


def _labels_from_wout(wout_path: Path, atom_symbols: List[str], atom_elem_ord: List[int], num_wann: int) -> Optional[List[str]]:
    """
    Heuristic parser for seedname.wout to get per-WF projection labels.
    Works only if the .wout contains a clear table with WF/projection indices.
    If it cannot build a full list of length num_wann, returns None.
    """
    lines = wout_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    labels: List[Optional[str]] = [None] * num_wann

    def atom_base(global_idx: int) -> str:
        if 1 <= global_idx <= len(atom_symbols):
            sym = atom_symbols[global_idx - 1]
            ord_ = atom_elem_ord[global_idx - 1]
            return f"{sym}{ord_}"
        return f"Atom{global_idx}"

    for line in lines:
        s = line.strip()
        if not s:
            continue

        idx = None
        m = re.match(r"^\|\s*(\d+)\s*\|", s)
        if m:
            idx = int(m.group(1))
        else:
            m = re.match(r"^(\d+)\s+", s)
            if m:
                idx = int(m.group(1))
            else:
                m = re.search(r"\bWF\s*(\d+)\b", s, flags=re.IGNORECASE)
                if m:
                    idx = int(m.group(1))

        if idx is None or not (1 <= idx <= num_wann):
            continue

        orb = _extract_orbital_keyword(s)
        if orb is None:
            continue

        atom_idx = None
        mA = re.search(r"Atom\s*:\s*(\d+)", s, flags=re.IGNORECASE)
        if mA:
            atom_idx = int(mA.group(1))

        # element symbol might appear like (Fe)
        elem = None
        mE = re.search(r"\(\s*([A-Za-z]{1,2})\s*\)", s)
        if mE:
            elem = mE.group(1)

        if atom_idx is not None:
            base = atom_base(atom_idx)
        elif elem is not None:
            base = elem
        else:
            base = f"WF{idx}"

        labels[idx - 1] = _sanitize_label(f"{base}_{orb}")

    if all(l is not None for l in labels):
        return [str(l) for l in labels]
    return None


def get_wannier_labels(num_wann: int, hr_path: Path, poscar_path: Path) -> List[str]:
    """
    Return a human-readable label list (length=num_wann) for Wannier orbitals.
    Priority:
      1) User-provided WANNIER_LABELS (exact length)
      2) AUTO_WANNIER_LABELS: parse <seedname>.wout (if possible) else <seedname>.win + POSCAR expansion
      3) fallback: w1, w2, ...
    """
    if WANNIER_LABELS is not None:
        if len(WANNIER_LABELS) == num_wann:
            print("[INFO] Using user-provided WANNIER_LABELS.")
            return list(WANNIER_LABELS)
        print(f"[WARN] WANNIER_LABELS length={len(WANNIER_LABELS)} != num_wann={num_wann}. Ignoring.")

    if not AUTO_WANNIER_LABELS:
        return [f"w{i+1}" for i in range(num_wann)]

    seed = SEEDNAME if SEEDNAME else _infer_seedname_from_hr(hr_path)
    win_path = Path(WIN_FILE) if WIN_FILE else hr_path.with_name(f"{seed}.win")
    wout_path = Path(WOUT_FILE) if WOUT_FILE else hr_path.with_name(f"{seed}.wout")

    try:
        atom_symbols, atom_elem_ord, _, _ = _read_poscar_atom_list(poscar_path)
    except Exception as e:
        print(f"[WARN] Failed to parse POSCAR atom list for auto labels: {e}")
        atom_symbols, atom_elem_ord = [], []

    # Try .wout first (most specific), then .win
    labels: Optional[List[str]] = None
    if wout_path.exists() and atom_symbols:
        labels = _labels_from_wout(wout_path, atom_symbols, atom_elem_ord, num_wann)
        if labels is not None:
            print(f"[INFO] Auto labels from WOUT: {wout_path} (len={len(labels)})")

    if labels is None and win_path.exists() and atom_symbols:
        labels_win = _labels_from_win(win_path, atom_symbols, atom_elem_ord, num_wann)
        if labels_win:
            print(f"[INFO] Auto labels from WIN: {win_path} (expanded len={len(labels_win)})")
            labels = labels_win

    if labels is None:
        print("[WARN] Could not auto-build Wannier labels (missing/parse failure of .win/.wout). Using w1,w2,...")
        return [f"w{i+1}" for i in range(num_wann)]

    # Pad / truncate
    if len(labels) < num_wann:
        print(f"[WARN] Auto labels produced {len(labels)} < num_wann={num_wann}. Padding with generic labels.")
        labels = labels + [f"w{i+1}" for i in range(len(labels), num_wann)]
    elif len(labels) > num_wann:
        print(f"[WARN] Auto labels produced {len(labels)} > num_wann={num_wann}. Truncating.")
        labels = labels[:num_wann]

    return labels



# ----------------------------------------------------------------------------
# NEW v13: automatic k0 selection along a k-line by minimizing |v|
# ----------------------------------------------------------------------------

def _read_win_bands_num_points(win_path: Union[str, Path]) -> Optional[int]:
    """
    Parse 'bands_num_points' from a wannier90.win file.
    Returns None if missing/unreadable.
    """
    try:
        txt = Path(win_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for line in txt.splitlines():
        # strip comments
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        if s.lower().startswith("bands_num_points"):
            # supports: "bands_num_points 101" or "bands_num_points = 101"
            nums = re.findall(r"[-+]?\d+", s)
            if nums:
                try:
                    n = int(nums[0])
                    if n >= 3:
                        return n
                except Exception:
                    pass
    return None


def scan_kline_minabs_v(
    lattice: Lattice,
    R_list: np.ndarray,
    degeneracy: np.ndarray,
    H_R: np.ndarray,
    u_hat_cart: np.ndarray,
    band_n_1based: int,
    k_start_frac: Sequence[float],
    k_end_frac: Sequence[float],
    num_points: int,
    track_by_overlap: bool = True,
    exclude_endpoints: bool = True,
    refine_root: bool = True,
    root_max_iters: int = 80,
    root_tol_t: float = 1e-12,
) -> Dict[str, object]:
    """
    Scan along a k-line (in fractional coordinates) and find k* that minimizes |v|, where
        v = dE/dk_u = <n|dH/dk_u|n>
    computed from the HR model.

    If refine_root=True and a sign change in v is detected between sample points, do a
    bisection refinement to locate v≈0 (more accurate than grid minimum).

    Returns:
        dict with keys:
          - k_star (tuple[float,float,float])
          - t_star (float), E_star (float), v_star (float), abs_v_star (float)
          - band_index_star (int, 1-based), overlap_star (float)
          - used_root_refine (bool)
          - table_rows (list of dict) for CSV export
    """
    k0 = np.array(k_start_frac, dtype=float)
    k1 = np.array(k_end_frac, dtype=float)
    dk = (k1 - k0)

    if num_points < 3:
        raise ValueError("num_points must be >= 3")

    t_grid = np.linspace(0.0, 1.0, num_points)

    # Precompute segment start in cart for kdist (Å^-1)
    k0_cart = k0 @ lattice.B

    # storage
    E_list: List[float] = []
    v_list: List[float] = []
    abs_v_list: List[float] = []
    idx_list: List[int] = []
    ov_list: List[float] = []
    vec_list: List[np.ndarray] = []
    kfrac_list: List[np.ndarray] = []
    kdist_list: List[float] = []

    prev_vec: Optional[np.ndarray] = None

    def _eval_at_k(k_frac: np.ndarray, vec_ref: Optional[np.ndarray]) -> Tuple[float, float, int, float, np.ndarray]:
        phase, w0, w1, w2, dotR = build_weights_and_dotR(
            lattice=lattice,
            R_list=R_list,
            degeneracy=degeneracy,
            k_frac=k_frac,
            u_hat_cart=u_hat_cart,
        )
        Hk, D1, D2 = build_H_D1_D2(w0, w1, w2, H_R, scale_override=None)
        evals, evecs = np.linalg.eigh(Hk)

        if (vec_ref is None) or (not track_by_overlap):
            idx = int(band_n_1based) - 1
            ov = 1.0
        else:
            overlaps = np.abs(evecs.conj().T @ vec_ref)
            idx = int(np.argmax(overlaps))
            ov = float(overlaps[idx])

        vn = evecs[:, idx]
        En = float(np.real(evals[idx]))
        v = float(np.real(np.vdot(vn, D1 @ vn)))
        return En, v, idx + 1, ov, vn

    # --- coarse scan ---
    for t in t_grid:
        k_frac = k0 + t * dk
        En, v, idx1, ov, vn = _eval_at_k(k_frac, prev_vec)
        prev_vec = vn

        k_cart = k_frac @ lattice.B
        kdist = float(np.linalg.norm(k_cart - k0_cart))

        E_list.append(En)
        v_list.append(v)
        abs_v_list.append(abs(v))
        idx_list.append(idx1)
        ov_list.append(ov)
        vec_list.append(vn)
        kfrac_list.append(k_frac.copy())
        kdist_list.append(kdist)

    # Choose discrete minimum
    cand = range(num_points)
    if exclude_endpoints and num_points > 2:
        cand = range(1, num_points - 1)

    j_min = min(cand, key=lambda j: abs_v_list[j])
    t_star = float(t_grid[j_min])
    k_star = kfrac_list[j_min]
    E_star = float(E_list[j_min])
    v_star = float(v_list[j_min])
    abs_v_star = float(abs_v_list[j_min])
    idx_star = int(idx_list[j_min])
    ov_star = float(ov_list[j_min])
    used_root = False

    # --- optional root refinement (v≈0) ---
    if refine_root and (num_points >= 3):
        # detect sign-change intervals (coarse)
        intervals: List[Tuple[int, int]] = []
        for j in range(num_points - 1):
            vL = v_list[j]
            vR = v_list[j + 1]
            if vL == 0.0:
                intervals.append((j, j))
            elif vL * vR < 0.0:
                intervals.append((j, j + 1))

        if intervals:
            # Prefer the interval closest to the discrete min point (in t)
            def _interval_score(pair: Tuple[int, int]) -> Tuple[float, float]:
                a, b = pair
                # if exact zero
                if a == b:
                    return (0.0, abs_v_list[a])
                tm = 0.5 * (t_grid[a] + t_grid[b])
                return (abs(tm - t_star), min(abs(v_list[a]), abs(v_list[b])))

            iL, iR = sorted(intervals, key=_interval_score)[0]
            if iL == iR:
                # already have v==0 on grid
                used_root = True
                j_min = iL
                t_star = float(t_grid[j_min])
                k_star = kfrac_list[j_min]
                E_star = float(E_list[j_min])
                v_star = float(v_list[j_min])
                abs_v_star = float(abs_v_list[j_min])
                idx_star = int(idx_list[j_min])
                ov_star = float(ov_list[j_min])
            else:
                # Bisection in t between iL and iR
                tL = float(t_grid[iL]); tR = float(t_grid[iR])
                vL = float(v_list[iL]); vR = float(v_list[iR])
                vecL = vec_list[iL]
                vecR = vec_list[iR]

                # Ensure bracket
                if vL * vR < 0.0:
                    for _it in range(int(root_max_iters)):
                        tM = 0.5 * (tL + tR)
                        kM = k0 + tM * dk
                        # Use the closer endpoint as reference for overlap tracking
                        vec_ref = vecL if (tM - tL) <= (tR - tM) else vecR
                        EM, vM, idxM, ovM, vecM = _eval_at_k(kM, vec_ref)

                        if (vL * vM) <= 0.0:
                            tR, vR, vecR = tM, vM, vecM
                        else:
                            tL, vL, vecL = tM, vM, vecM

                        if abs(tR - tL) < float(root_tol_t) or abs(vM) < 1e-12:
                            break

                    t_star = 0.5 * (tL + tR)
                    k_star = k0 + t_star * dk
                    # Evaluate final at k_star with ref vecL
                    E_star, v_star, idx_star, ov_star, _vec = _eval_at_k(k_star, vecL)
                    abs_v_star = abs(v_star)
                    used_root = True

    # Build table rows for CSV
    rows: List[Dict[str, object]] = []
    for j in range(num_points):
        kf = kfrac_list[j]
        rows.append({
            "j": j,
            "t": float(t_grid[j]),
            "kdist_seg": float(kdist_list[j]),
            "kx_frac": float(kf[0]),
            "ky_frac": float(kf[1]),
            "kz_frac": float(kf[2]),
            "band_index_tracked": int(idx_list[j]),
            "overlap": float(ov_list[j]),
            "E_eV": float(E_list[j]),
            "v_eVAng": float(v_list[j]),
            "abs_v_eVAng": float(abs_v_list[j]),
        })

    # Append refined root as an extra row (optional) if not on-grid
    if used_root:
        # check if t_star coincides with a grid point
        if np.min(np.abs(t_grid - t_star)) > 1e-14:
            k_cart = k_star @ lattice.B
            kdist = float(np.linalg.norm(k_cart - k0_cart))
            rows.append({
                "j": "root",
                "t": float(t_star),
                "kdist_seg": float(kdist),
                "kx_frac": float(k_star[0]),
                "ky_frac": float(k_star[1]),
                "kz_frac": float(k_star[2]),
                "band_index_tracked": int(idx_star),
                "overlap": float(ov_star),
                "E_eV": float(E_star),
                "v_eVAng": float(v_star),
                "abs_v_eVAng": float(abs_v_star),
            })

    return {
        "k_star": (float(k_star[0]), float(k_star[1]), float(k_star[2])),
        "t_star": float(t_star),
        "E_star": float(E_star),
        "v_star": float(v_star),
        "abs_v_star": float(abs_v_star),
        "band_index_star": int(idx_star),
        "overlap_star": float(ov_star),
        "used_root_refine": bool(used_root),
        "table_rows": rows,
    }

def main():
    _normalize_runtime_params()
    # ------------------ Load inputs ------------------
    hr_path = Path(HR_FILE)
    poscar_path = Path(POSCAR_FILE)
    if not hr_path.exists():
        raise SystemExit(f"ERROR: HR_FILE not found: {hr_path}")
    if not poscar_path.exists():
        raise SystemExit(f"ERROR: POSCAR_FILE not found: {poscar_path}")

    # Lattice vectors are ONLY used for derivatives (dotR in Å).
    # If POSCAR and the Wannier90 unit cell differ, energies will still match but curvature will be wrong.
    seed = SEEDNAME if SEEDNAME else _infer_seedname_from_hr(hr_path)
    win_path = Path(WIN_FILE) if WIN_FILE else hr_path.with_name(f"{seed}.win")
    lattice, lattice_src = get_lattice_from_inputs(poscar_path, win_path, LATTICE_SOURCE)
    lens = np.linalg.norm(lattice.A, axis=1)
    print(f"[INFO] Lattice source for curvature: {lattice_src}")
    print(f"       |a1|,|a2|,|a3| = {lens.tolist()} Å")
    print("")
    num_wann, nrpts, R_list, degeneracy, H_R = read_wannier90_hr_dat(hr_path)
    labels = get_wannier_labels(num_wann, hr_path, poscar_path)
    if EXPORT_WANNIER_LABELS:
        write_csv(
            "wannier_labels_used.csv",
            fieldnames=["wf", "label"],
            rows=[{"wf": i + 1, "label": labels[i]} for i in range(num_wann)],
        )
        print("[OUT] Wrote Wannier labels used to: wannier_labels_used.csv")
        print("")

    k0_user = tuple(float(x) for x in K_FRAC)
    k0 = k0_user
    u_hat, mode_used = build_direction_unit(
        lattice=lattice,
        dir_mode=DIR_MODE,
        dir_cart=DIR_CART,
        kline_start=KLINE_START,
        kline_end=KLINE_END,
    )

    # ------------------ Optional: auto-select k0 by minimizing |v| along a k-line ------------------
    if AUTO_K0_MINABS_V_ENABLE:
        # Determine scan segment
        kscan_start = np.array(KSCAN_START if KSCAN_START is not None else KLINE_START, dtype=float)
        kscan_end   = np.array(KSCAN_END   if KSCAN_END   is not None else KLINE_END,   dtype=float)

        # Resolve number of scan points
        n_scan = None
        if isinstance(KSCAN_NUM_POINTS, str) and KSCAN_NUM_POINTS.lower() == "win":
            # use bands_num_points from the win file (if available)
            try:
                n_scan = _read_win_bands_num_points(win_path)
            except Exception:
                n_scan = None
            if n_scan is None:
                n_scan = 101
        else:
            n_scan = int(KSCAN_NUM_POINTS)

        scan_res = scan_kline_minabs_v(
            lattice=lattice,
            R_list=R_list,
            degeneracy=degeneracy,
            H_R=H_R,
            u_hat_cart=u_hat,
            band_n_1based=BAND_N,
            k_start_frac=kscan_start.tolist(),
            k_end_frac=kscan_end.tolist(),
            num_points=n_scan,
            track_by_overlap=KSCAN_TRACK_BY_OVERLAP,
            exclude_endpoints=KSCAN_EXCLUDE_ENDPOINTS,
            refine_root=KSCAN_REFINE_ROOT,
            root_max_iters=KSCAN_ROOT_MAX_ITERS,
            root_tol_t=KSCAN_ROOT_TOL_T,
        )

        k_star = scan_res["k_star"]
        print("=== Auto k0 selection by min |v| (group velocity) ===")
        print(f"Scan segment (frac): start={kscan_start.tolist()}  end={kscan_end.tolist()}  n={n_scan}")
        print(f"Band tracked from BAND_N={BAND_N} (1-based), overlap_tracking={KSCAN_TRACK_BY_OVERLAP}")
        print(f"Selected k*: {list(k_star)}   t={scan_res['t_star']:.12f}")
        print(f"  E(k*) = {scan_res['E_star']:+.10f} eV")
        print(f"  v(k*) = {scan_res['v_star']:+.6e} eV·Å   |v|={scan_res['abs_v_star']:.6e}")
        print(f"  tracked_band_index = {scan_res['band_index_star']}  overlap={scan_res['overlap_star']:.6f}")
        if scan_res.get("used_root_refine", False):
            print("  (root refinement used: v≈0)")
        print("")

        if EXPORT_KSCAN_TABLE:
            write_csv(
                KSCAN_TABLE_CSV,
                fieldnames=["j","t","kdist_seg","kx_frac","ky_frac","kz_frac","band_index_tracked","overlap","E_eV","v_eVAng","abs_v_eVAng"],
                rows=scan_res["table_rows"],
            )
            print(f"[OUT] Wrote k-line scan table to: {KSCAN_TABLE_CSV}")
            print("")

        if AUTO_K0_MINABS_V_USE_FOR_ANALYSIS:
            k0 = tuple(float(x) for x in k_star)


    # weights
    phase, w0, w1, w2, dotR = build_weights_and_dotR(
        lattice=lattice, R_list=R_list, degeneracy=degeneracy, k_frac=k0, u_hat_cart=u_hat
    )

    print("=== Inputs ===")
    print(f"hr      : {hr_path}")
    print(f"POSCAR  : {poscar_path}")
    print(f"num_wann: {num_wann}, nrpts: {nrpts}")
    if tuple(k0) == tuple(k0_user):
        print(f"k_frac  : {list(k0)}")
    else:
        print(f"k_frac (user) : {list(k0_user)}")
        print(f"k_frac (used) : {list(k0)}")
    print(f"band_n  : {BAND_N} (1-based)")
    print(f"band_m  : {BAND_M} (1-based)")
    print(f"TOPN    : {TOPN_BANDS}")
    print("")
    print(f"Direction mode: {mode_used}")
    print(f"dir_cart (unit): {u_hat.tolist()}")
    print("")

    # ------------------ Baseline H, D1, D2 ------------------
    Hk0, D10, D20 = build_H_D1_D2(w0, w1, w2, H_R, scale_override=None)
    base = curvature_and_interband_table(Hk0, D10, D20, band_n_1based=BAND_N, band_m_1based=BAND_M)
    evals0: np.ndarray = base["evals"]
    evecs0: np.ndarray = base["evecs"]
    En0 = float(base["En"])
    C0_intra = float(base["C_intra"])
    C0_inter = float(base["C_inter"])
    C0_total = float(base["C_total"])

    print("=== Results (units) ===")
    print("H in eV; dH/dk_u in eV·Å; |V|^2 in (eV·Å)^2; curvature in eV·Å^2")
    print("")
    print(f"Band n = {BAND_N}, E_n = {En0:.10f} eV")

    pair = base.get("pair", None)
    if pair is not None:
        print(f"Band m = {pair['m']}, E_m = {pair['Em']:.10f} eV")
        print(f"ΔE = E_n - E_m = {pair['dE']:.10e} eV")
        print(f"|<n|dH/dk_u|m>|^2 = {pair['V2']:.10e} (eV·Å)^2")
        print(f"Pair contribution 2|V|^2/ΔE = {pair['contrib']:.10e} eV·Å^2")
        print("")

    print("=== Curvature decomposition for band n ===")
    if abs(C0_total) > 1e-14:
        pintra = 100.0 * C0_intra / C0_total
        pinter = 100.0 * C0_inter / C0_total
    else:
        pintra = float("nan")
        pinter = float("nan")
    print(f"Intraband  <n|d²H/dk_u²|n>              : {C0_intra:.10e} eV·Å^2   ({pintra:.2f} % of total)")
    print(f"Interband  2 Σ_m≠n |V|^2/(E_n-E_m)      : {C0_inter:.10e} eV·Å^2   ({pinter:.2f} % of total)")
    print(f"Total curvature (intra + inter)         : {C0_total:.10e} eV·Å^2")
    if abs(C0_total) > 1e-14:
        mstar = HBAR2_OVER_ME / C0_total
        print(f"Estimated effective mass m*/m_e         : {mstar:.6f}  (ħ²/m_e = {HBAR2_OVER_ME} eV·Å²)")
    else:
        print("Estimated effective mass m*/m_e         : undefined (C_total≈0)")
    print("")

    # ------------------ Interband TOP-N + export baseline ------------------
    inter_rows = base["inter_table"]
    inter_sorted = sorted(inter_rows, key=lambda r: -abs(float(r["contrib"])))
    print(f"Top-{TOPN_BANDS} interband contributions (sorted by |2|V|^2/ΔE|):")
    print("  m   E_m(eV)      ΔE(eV)       |V_nm|^2((eV·Å)^2)   2|V|^2/ΔE(eV·Å^2)")
    for r in inter_sorted[:min(TOPN_BANDS, len(inter_sorted))]:
        print(f"{r['m']:>4d} {r['Em']:>12.10f} {r['dE']:>14.10e} {r['V2']:>18.10e} {r['contrib']:>18.10e}")
    print("")

    if EXPORT_FULL_INTERBAND_TABLE:
        out_rows = []
        for r in inter_sorted:
            out_rows.append({"case": "baseline", "group": "", "lambda": 1.0, **r})
        write_csv(
            "interband_contrib_baseline.csv",
            fieldnames=["case", "group", "lambda", "m", "Em", "dE", "V2", "contrib"],
            rows=out_rows,
        )
        print("[OUT] Wrote full baseline interband table to: interband_contrib_baseline.csv")
        print("")

    # ------------------ Optional: numerical curvature check vs band.dat ------------------
    bandcheck_rows: List[Dict[str, object]] = []
    if BAND_CHECK_ENABLE:
        bandcheck_rows = run_band_curvature_check(
            hr_path=hr_path,
            band_n_1based=(BAND_CHECK_BAND_INDEX or BAND_N),
            C_analytic=C0_total,
            u_mode=mode_used,
        )

    # ------------------ Optional: HR numerical derivative/curvature self-check (recommended) ------------------
    if HR_NUM_CHECK_ENABLE:
        _ = run_hr_numerical_check(
            lattice=lattice,
            R_list=R_list,
            degeneracy=degeneracy,
            H_R=H_R,
            k0_frac=k0,
            u_hat_cart=u_hat,
            band_n_1based=BAND_N,
            Hk0=Hk0,
            D10=D10,
            D20=D20,
            evals0=evals0,
            evecs0=evecs0,
            C_analytic=C0_total,
            bandcheck_rows=bandcheck_rows,
        )

    # ------------------ Intraband per-R contributions (scalar) ------------------
    if EXPORT_INTRA_PER_R:
        # scalar per-R C_intra contribution
        rows = []
        for r in range(nrpts):
            D2R = w2[r] * H_R[r]
            vn = evecs0[:, BAND_N - 1]
            contrib = float(np.real(np.vdot(vn, D2R @ vn)))
            rows.append({
                "R1": int(R_list[r, 0]),
                "R2": int(R_list[r, 1]),
                "R3": int(R_list[r, 2]),
                "contrib_intra": contrib,
                "weight_Re": float(np.real(w2[r])),
                "weight_Im": float(np.imag(w2[r])),
                "dotR_Ang": float(dotR[r]),
                "deg": int(degeneracy[r]),
            })
        sum_perR = sum(float(rr["contrib_intra"]) for rr in rows)
        print("=== Intraband per-R contributions (for pre-screen) ===")
        print(f"Sum_R contrib_R = {sum_perR:.10e} eV·Å^2")
        print(f"C_intra (from full D2) = {C0_intra:.10e} eV·Å^2")
        print(f"Difference (sum - C_intra) = {sum_perR - C0_intra:+.3e} eV·Å^2  (numerical noise expected)")
        print("")
        write_csv(
            "intra_contrib_by_R.csv",
            fieldnames=["R1", "R2", "R3", "contrib_intra", "weight_Re", "weight_Im", "dotR_Ang", "deg"],
            rows=rows,
        )
        print("[OUT] Wrote per-R intraband contributions to: intra_contrib_by_R.csv")
        print("")

    # ------------------ Harmonic grouping by q·R ------------------
    dk = np.array(KLINE_END, dtype=float) - np.array(KLINE_START, dtype=float)
    q, D, fracs = rationalize_delta_k(dk.tolist(), DENOM_MAX)
    labels_g, group_to_indices, n_raw, abs_n = harmonic_group_labels(R_list, q, MAX_ABS_N)

    print("=== Harmonic grouping (n(R)=q·R, group=|n|) ===")
    print(f"k_line_start: {list(KLINE_START)}")
    print(f"k_line_end  : {list(KLINE_END)}")
    print(f"Δk          : {dk.tolist()}  ~  {[str(f) for f in fracs]}")
    print(f"q           : {q.tolist()},  D={D}")
    print("")
    print("Group meaning (IMPORTANT): group g contains all R with |q·R| = g.")
    if q.tolist() == [0, 1, 0]:
        print("  For this case q=[0,1,0] -> n(R)=R2:")
        print("    group 0: R2=0;  group 1: R2=±1;  group 2: R2=±2; ...")
    print("")

    groups = sorted(group_to_indices.keys(), key=lambda x: (999999 if isinstance(x, str) else int(x)))
    print(f"Total groups: {len(groups)} (including |n|=0)")
    print("")

    # Aggregate intra group scores (using scalar per-R contributions from above)
    # Here we recompute quickly from w2 and vn to ensure availability even if EXPORT_INTRA_PER_R=False
    vn0 = evecs0[:, BAND_N - 1]
    perR_scalar = np.zeros(nrpts, dtype=float)
    for r in range(nrpts):
        perR_scalar[r] = float(np.real(np.vdot(vn0, (w2[r] * H_R[r]) @ vn0)))

    group_stats = []
    for g in groups:
        idxs = group_to_indices[g]
        s = float(np.sum(perR_scalar[idxs]))
        a = float(np.sum(np.abs(perR_scalar[idxs])))
        group_stats.append({"group": g, "sum": s, "abs_sum": a, "nR": int(len(idxs))})
    group_stats_sorted = sorted(group_stats, key=lambda x: -float(x["abs_sum"]))

    print("Top groups by abs_sum (intraband pre-screen):")
    for gs in group_stats_sorted[:min(10, len(group_stats_sorted))]:
        print(f"  group {str(gs['group']):>6}  abs_sum={gs['abs_sum']:.6e}  sum={gs['sum']:+.6e}  nR={gs['nR']}")
    print("")

    write_csv(
        "intra_group_scores.csv",
        fieldnames=["group", "sum", "abs_sum", "nR"],
        rows=[{"group": str(gs["group"]), "sum": gs["sum"], "abs_sum": gs["abs_sum"], "nR": gs["nR"]} for gs in group_stats_sorted],
    )
    print("[OUT] Wrote group score table to: intra_group_scores.csv")
    print("")

    # ------------------ Select groups for ablation (v14 flexible selection) ------------------
    # Print how many harmonic groups are actually present in the HR file.
    # (If max |n| present is only 2, then groups 3/4/5 cannot contribute because they are absent in HR.)
    int_groups_present = [int(g) for g in groups if isinstance(g, (int, np.integer))]
    max_g_present = max(int_groups_present) if int_groups_present else None
    if max_g_present is not None:
        print(f"[INFO] Max integer group present in HR: {max_g_present}")
    else:
        print(f"[INFO] No integer harmonic groups detected (unexpected).")

    mode = str(ABLATE_GROUP_MODE).lower().strip()
    if mode == "top":
        selected = [gs["group"] for gs in group_stats_sorted[:min(TOP_GROUPS_FOR_ABLATION, len(group_stats_sorted))]]
        sel_msg = f"top {TOP_GROUPS_FOR_ABLATION} by abs_sum"
    elif mode == "upto":
        selected = list(range(0, int(ABLATE_GROUP_MAX) + 1))
        sel_msg = f"all integer groups g=0..{int(ABLATE_GROUP_MAX)}"
    elif mode == "list":
        selected = list(ABLATE_GROUP_LIST)
        sel_msg = f"explicit list {selected}"
    elif mode == "all":
        selected = list(groups)
        sel_msg = "all groups present in HR"
    else:
        raise ValueError(f"Unknown ABLATE_GROUP_MODE='{ABLATE_GROUP_MODE}'. Use 'top'/'upto'/'list'/'all'.")

    # Ensure group_to_indices has entries for all selected groups (even if nR=0).
    for g in selected:
        if g not in group_to_indices:
            group_to_indices[g] = []

    # If user requested groups that are not present in HR, warn clearly.
    missing = [g for g in selected if (isinstance(g, (int, np.integer)) and max_g_present is not None and int(g) > int(max_g_present))]
    if missing:
        print(f"[WARN] Requested groups {missing} exceed max group present ({max_g_present}). They have nR=0 in this HR.")

    print(f"Selected groups for ablation ({sel_msg}): {selected}")
    print(f"Ablation scaling λ = {LAMBDA_ABLATE}")
    print("")

    # ------------------ NEW: Orbital-resolved analysis within each selected group ------------------
    if EXPORT_ORBPAIR_GROUP_RANKING:
        print("=== Orbital-resolved (Wannier-pair) attribution within each group ===")

        # Baseline D1 eigen-basis row V_nm for building inter sensitivity vector S
        D1_eig0 = evecs0.conj().T @ D10 @ evecs0  # (nb,nb)
        n_idx = BAND_N - 1
        Vrow = D1_eig0[n_idx, :]  # <n|D1|m>
        denom = (En0 - evals0).real.astype(float)

        # build A_m = conj(V_nm)/(En-Em), skipping m=n
        A = np.zeros_like(Vrow, dtype=np.complex128)
        for m in range(Vrow.size):
            if m == n_idx:
                continue
            if INTER_SENS_M_LIST and (m + 1) not in set(INTER_SENS_M_LIST):
                continue
            A[m] = np.conj(Vrow[m]) / denom[m]

        # S_j = Σ_m A_m * (vm)_j = (evecs0 @ A)_j
        S = evecs0 @ A  # (num_wann,)

        # Precompute group matrices M1_g and M2_g
        # M1_g = Σ_{R in g} w1(R) H(R)
        # M2_g = Σ_{R in g} w2(R) H(R)
        for g in selected:
            idxs = group_to_indices.get(g, [])
            outname = f"orbpair_group_{str(g)}_ranking.csv"

            # If a group is not present in HR (nR=0), we still create an empty file for completeness.
            if len(idxs) == 0:
                write_csv(
                    outname,
                    fieldnames=["group", "i", "j", "label_i", "label_j",
                                "intra_contrib", "inter_sens", "total_sens",
                                "abs_intra", "abs_inter", "abs_total"],
                    rows=[],
                )
                print(f"[OUT] {outname}  (group not present in HR; nR=0)")
                continue

            M1g = np.tensordot(w1[idxs], H_R[idxs], axes=(0, 0))
            M2g = np.tensordot(w2[idxs], H_R[idxs], axes=(0, 0))

            # orbital-pair resolved derivatives / contributions at fixed eigenvectors
            intra_mat = np.real(vn0.conj()[:, None] * M2g * vn0[None, :])
            inter_sens_mat = 4.0 * np.real(vn0.conj()[:, None] * M1g * S[None, :])
            total_sens_mat = intra_mat + inter_sens_mat

            # build ranking rows
            rows_rank = []
            for i in range(num_wann):
                for j in range(num_wann):
                    intra_ij = float(intra_mat[i, j])
                    inter_ij = float(inter_sens_mat[i, j])
                    total_ij = float(total_sens_mat[i, j])
                    rows_rank.append({
                        "group": str(g),
                        "i": i + 1,
                        "j": j + 1,
                        "label_i": labels[i],
                        "label_j": labels[j],
                        "intra_contrib": intra_ij,
                        "inter_sens": inter_ij,
                        "total_sens": total_ij,
                        "abs_intra": abs(intra_ij),
                        "abs_inter": abs(inter_ij),
                        "abs_total": abs(total_ij),
                    })

            # sort
            if ORBPAIR_RANK_SORT == "abs_intra":
                keyf = lambda r: -float(r["abs_intra"])
            elif ORBPAIR_RANK_SORT == "abs_inter":
                keyf = lambda r: -float(r["abs_inter"])
            else:
                keyf = lambda r: -float(r["abs_total"])
            rows_rank.sort(key=keyf)

            outname = f"orbpair_group_{str(g)}_ranking.csv"
            write_csv(
                outname,
                fieldnames=["group", "i", "j", "label_i", "label_j",
                            "intra_contrib", "inter_sens", "total_sens",
                            "abs_intra", "abs_inter", "abs_total"],
                rows=rows_rank[:min(TOP_ORBPAIRS_PER_GROUP, len(rows_rank))],
            )
            print(f"[OUT] {outname}  (top {TOP_ORBPAIRS_PER_GROUP} pairs by {ORBPAIR_RANK_SORT})")
        print("")

    # ------------------ NEW: Orbital-pair contributions for each exact R (C_intra only) ------------------
    if EXPORT_ORBPAIR_BY_R_TOP:
        rows_topR = []
        for r in range(nrpts):
            R1, R2, R3 = int(R_list[r, 0]), int(R_list[r, 1]), int(R_list[r, 2])
            # group label (|q·R|) using already computed abs_n array
            gR = labels_g[r]
            # matrix contribution for this R to C_intra (fixed vn)
            M2R = w2[r] * H_R[r]
            intra_mat_R = np.real(vn0.conj()[:, None] * M2R * vn0[None, :])

            # find top pairs
            flat = intra_mat_R.reshape(-1)
            idx_sorted = np.argsort(-np.abs(flat))
            topk = min(TOP_ORBPAIRS_PER_R, flat.size)
            for rank in range(topk):
                idx = int(idx_sorted[rank])
                i = idx // num_wann
                j = idx % num_wann
                val = float(intra_mat_R[i, j])
                Hij = H_R[r, i, j]
                rows_topR.append({
                    "R1": R1, "R2": R2, "R3": R3,
                    "group": str(gR),
                    "rank": rank + 1,
                    "i": i + 1, "j": j + 1,
                    "label_i": labels[i], "label_j": labels[j],
                    "contrib_intra_ij": val,
                    "abs_contrib": abs(val),
                    "H_re": float(np.real(Hij)),
                    "H_im": float(np.imag(Hij)),
                    "absH": float(np.abs(Hij)),
                    "weight2_Re": float(np.real(w2[r])),
                    "weight2_Im": float(np.imag(w2[r])),
                    "dotR_Ang": float(dotR[r]),
                    "deg": int(degeneracy[r]),
                })

        write_csv(
            "orbpair_by_R_top.csv",
            fieldnames=["R1", "R2", "R3", "group", "rank",
                        "i", "j", "label_i", "label_j",
                        "contrib_intra_ij", "abs_contrib",
                        "H_re", "H_im", "absH",
                        "weight2_Re", "weight2_Im", "dotR_Ang", "deg"],
            rows=rows_topR,
        )
        print("[OUT] Wrote top orbital-pair contributions per R to: orbpair_by_R_top.csv")
        print("")
        if MERGE_ORBPAIR_BY_R_TOP_HERMITIAN:
            merged_rows = merge_orbpair_by_R_top_rows(
                rows_topR,
                use_dotR=MERGE_ORBPAIR_CANONICAL_USE_DOTR,
                dotR_eps=MERGE_ORBPAIR_DOTR_EPS,
                shortlabel=False,
            )
            write_csv(
                ORBPAIR_BY_R_TOP_MERGED_FILE,
                fieldnames=["R1", "R2", "R3", "group", "rank_merged",
                            "i", "j", "label_i", "label_j",
                            "n_terms", "pair_complete",
                            "contrib_sum", "contrib_mean", "abs_contrib_sum",
                            "H_re", "H_im", "absH",
                            "weight2_Re", "weight2_Im", "dotR_Ang", "deg",
                            "min_rank", "max_rank"],
                rows=merged_rows,
            )
            n_in = len(rows_topR)
            n_out = len(merged_rows)
            n_pair = sum(int(r.get("pair_complete", 0)) for r in merged_rows)
            print(f"[OUT] Wrote Hermitian-merged orbpair table to: {ORBPAIR_BY_R_TOP_MERGED_FILE}  (rows: {n_in} -> {n_out}, complete pairs: {n_pair})")

            if EXPORT_ORBPAIR_BY_R_TOP_MERGED_SHORTLABEL:
                merged_rows_short = merge_orbpair_by_R_top_rows(
                    rows_topR,
                    use_dotR=MERGE_ORBPAIR_CANONICAL_USE_DOTR,
                    dotR_eps=MERGE_ORBPAIR_DOTR_EPS,
                    shortlabel=True,
                )
                write_csv(
                    ORBPAIR_BY_R_TOP_MERGED_SHORTLABEL_FILE,
                    fieldnames=["R1", "R2", "R3", "group", "rank_merged",
                                "i", "j", "label_i", "label_j",
                                "label_i_short", "label_j_short",
                                "n_terms", "pair_complete",
                                "contrib_sum", "contrib_mean", "abs_contrib_sum",
                                "H_re", "H_im", "absH",
                                "weight2_Re", "weight2_Im", "dotR_Ang", "deg",
                                "min_rank", "max_rank"],
                    rows=merged_rows_short,
                )
                print(f"[OUT] Wrote Hermitian-merged (shortlabel) table to: {ORBPAIR_BY_R_TOP_MERGED_SHORTLABEL_FILE}")

            print("")


    # ======================================================================
        # Step A/B/C: 单个 hopping“旋钮”灵敏度 + P0(两份 HR)映射与验证 + 可选 toy 调参
    # ----------------------------------------------------------------------
    # 说明：
    #   Step A  : 在基态 HR 上计算每个 Hermitian-merged (R,i,j) 对曲率 C 的线性灵敏度 S = dC/dλ |_{λ=1}
    #   Step B  : (可选) 用 P0：从第二份 HR(物理扰动后的 Wannier)抽取每个 hopping 的 λ_p0，并给出一阶预测 ΔC_pred
    #   Step C  : (可选) 直接用第二份 HR 计算“真实”曲率 C_true，与 ΔC_pred 对比（验证线性近似覆盖度）
    #   Toy knob: (可选) 用户手动指定若干 hopping 的 λ，直接修改 HR 重新算 C（用于因果验证/可视化）
    # ----------------------------------------------------------------------

    if KNOB_SENS_ENABLE or KNOB_TUNE_ENABLE or P0_ENABLE:

        print("\n=== Knob sensitivity table (R,i,j Hermitian-merged) ===")

        # Baseline anchors used throughout
        k_frac_used = list(map(float, k0))
        band_n_idx = int(BAND_N - 1)  # 0-based

        # Baseline reference band vector (already computed): vn0
        # Baseline reference curvature: C0_total

        # ------------------ Optional P0: load perturbed HR and choose its (k,band) ------------------
        H_R_p0 = None
        R_list_p0 = None
        degeneracy_p0 = None
        lattice_p0 = None
        lattice_src_p0 = None
        k_frac_p0 = None
        band_n_p0_1based = None
        u_hat_p0 = None
        dir_cart_p0 = None
        R_to_idx_p0 = None

        if P0_ENABLE:
            # Read P0 HR
            R_list_p0, degeneracy_p0, H_R_p0 = read_wannier90_hr(HR_FILE_P0, NUM_WANN)
            R_to_idx_p0 = {tuple(map(int, R)): idx for idx, R in enumerate(R_list_p0)}

            # Lattice for P0 (can be different under strain)
            poscar_p0_path = POSCAR_P0 if POSCAR_P0 else POSCAR
            win_p0_path = WIN_FILE_P0 if WIN_FILE_P0 else WIN_FILE
            lattice_p0, lattice_src_p0 = get_lattice_from_inputs(poscar_p0_path, win_p0_path, LATTICE_SOURCE)

            # Direction unit vector for P0
            u_hat_p0, dir_cart_p0 = build_direction_unit(
                lattice=lattice_p0,
                dir_mode=DIR_MODE,
                dir_cart=np.array(DIR_CART, dtype=float),
                k_line_start=KLINE_START,
                k_line_end=KLINE_END,
            )

            # Choose k-point for P0
            mode = str(P0_KPOINT_MODE).lower().strip()
            if mode in ("each_minv", "each_minabs_v", "each"):
                print("\n=== P0 k0 selection (each structure by min |v|) ===")
                kscan_rows_p0, k_frac_p0, _, _, _, band_tracked_p0, _ = scan_kline_minabs_v(
                    R_list=R_list_p0,
                    degeneracy=degeneracy_p0,
                    H_R=H_R_p0,
                    lattice=lattice_p0,
                    k_start=KLINE_START,
                    k_end=KLINE_END,
                    npts=AUTO_K0_MINABS_V_NPTS,
                    band0_1based=BAND_N,
                    overlap_track=AUTO_K0_OVERLAP_TRACK,
                    refine_root=AUTO_K0_REFINE_ROOT,
                    dir_mode=DIR_MODE,
                    dir_cart=DIR_CART,
                )
                band_n_p0_1based = int(band_tracked_p0)
                if P0_KSCAN_CSV:
                    write_csv(
                        P0_KSCAN_CSV,
                        fieldnames=list(kscan_rows_p0[0].keys()) if kscan_rows_p0 else None,
                        rows=kscan_rows_p0,
                    )
            else:
                # same_k: compare at the same fractional k used for baseline
                k_frac_p0 = k_frac_used
                band_n_p0_1based = int(BAND_N)

                if P0_BAND_TRACK_BY_OVERLAP:
                    # Track band index at the same k by overlap with baseline eigenvector
                    _, w0_p0, w1_p0, w2_p0, dotR_tmp = build_weights_and_dotR(
                        lattice=lattice_p0,
                        R_list=R_list_p0,
                        degeneracy=degeneracy_p0,
                        k_frac=k_frac_p0,
                        u_hat=u_hat_p0,
                    )
                    Hk_p0, _, _ = build_H_D1_D2(w0_p0, w1_p0, w2_p0, H_R_p0)
                    evals_p0, evecs_p0 = np.linalg.eigh(Hk_p0)
                    band_n_p0_1based, ov = band_track_index(evecs_ref=evecs0, evecs_new=evecs_p0, idx_ref_1based=BAND_N)
                    print(f"[P0] Band overlap tracking @ same k: chosen_band={band_n_p0_1based}  overlap={ov:.6f}")

        # Build inter sensitivity vector S for knob analysis.
        # NOTE: S must be available regardless of whether EXPORT_ORBPAIR_GROUP_RANKING is enabled.
        D1_eig0_knob = evecs0.conj().T @ D10 @ evecs0  # (nb, nb)
        Vrow_knob = D1_eig0_knob[band_n_idx, :]        # <n|D1|m>
        denom_knob = (En0 - evals0).real.astype(float)

        A_knob = np.zeros_like(Vrow_knob, dtype=np.complex128)
        for m in range(Vrow_knob.size):
            if m == band_n_idx:
                continue
            if INTER_SENS_M_LIST and (m + 1) not in set(INTER_SENS_M_LIST):
                continue
            A_knob[m] = np.conj(Vrow_knob[m]) / denom_knob[m]
        S_knob = evecs0 @ A_knob  # (num_wann,)

        # ------------------ Step A/B: build knob sensitivity (+ optional P0 lambda) table ------------------
        knob_rows = compute_knob_table_Rij(
            R_list=R_list,
            degeneracy=degeneracy,
            H_R=H_R,
            w1=w1,
            w2=w2,
            dotR=dotR,
            group_labels=labels_g,
            q_vec_int=q,
            band_vec_n=vn0,
            S_vec=S_knob,
            labels=labels,
            group_max=KNOB_MAX_GROUP,
            min_abs_t0=KNOB_MIN_ABS_T0,
            eps_dotR=KNOB_DOTR_EPS,
            H_R_p0=H_R_p0,
            R_to_idx_p0=R_to_idx_p0,
        )

        if knob_rows:
            # Always export Step-A sensitivity table
            if KNOB_SENSITIVITY_CSV:
                if KNOB_EXPORT_FULL:
                    write_csv(KNOB_SENSITIVITY_CSV, fieldnames=list(knob_rows[0].keys()), rows=knob_rows)
                else:
                    # slim columns
                    keep_cols = [
                        "R1","R2","R3","i","j","label_i","label_j","dotR","group",
                        "S_intra","S_inter","S_total",
                    ]
                    if P0_ENABLE:
                        keep_cols += ["lam_p0","pred_dC_intra","pred_dC_inter","pred_dC_total"]
                    slim = [{k: r.get(k, "") for k in keep_cols} for r in knob_rows]
                    write_csv(KNOB_SENSITIVITY_CSV, fieldnames=keep_cols, rows=slim)

            # If P0 is enabled, also export the same table to a dedicated mapping CSV (optional)
            if P0_ENABLE and KNOB_P0_MAPPING_CSV and (KNOB_P0_MAPPING_CSV != KNOB_SENSITIVITY_CSV):
                write_csv(KNOB_P0_MAPPING_CSV, fieldnames=list(knob_rows[0].keys()), rows=knob_rows)

            # Print top-N knobs
            def _abs_float(x):
                try:
                    return abs(float(x))
                except Exception:
                    return 0.0

            knob_sorted = sorted(knob_rows, key=lambda r: _abs_float(r.get("S_total", 0.0)), reverse=True)
            print(f"[INFO] Knob rows kept: {len(knob_rows)}  (group<= {KNOB_MAX_GROUP}, |t0|>={KNOB_MIN_ABS_T0})")
            print(f"Top-{KNOB_TOPN_PRINT} knobs by |S_total|:")
            for rr in knob_sorted[: int(KNOB_TOPN_PRINT)]:
                s_intra = rr.get("S_intra", 0.0)
                s_inter = rr.get("S_inter", 0.0)
                s_total = rr.get("S_total", 0.0)
                info = f"R=({rr['R1']},{rr['R2']},{rr['R3']})  (i,j)=({rr['i']},{rr['j']})  {rr.get('label_i','')} ↔ {rr.get('label_j','')}"
                if P0_ENABLE:
                    lam_p0 = rr.get("lam_p0", "")
                    pred = rr.get("pred_dC_total", "")
                    print(f"  |S|={abs(float(s_total)):.3e}  S={float(s_total):+.3e}  lam_p0={lam_p0}  pred_dC={pred}   {info}")
                else:
                    print(f"  |S|={abs(float(s_total)):.3e}  S={float(s_total):+.3e}  (intra={float(s_intra):+.3e}, inter={float(s_inter):+.3e})   {info}")
        else:
            print("[WARN] No knob rows survived filters; consider increasing KNOB_MAX_GROUP or lowering KNOB_MIN_ABS_T0.")

        # ------------------ (B) Step-P1: on-site additive module ------------------
        if P1_ENABLE:
            run_P1_onsite_additive(
                H_R=H_R,
                R_list=R_list,
                degeneracy=degeneracy,
                lattice=lattice,
                k_frac=k_frac_used,
                u_hat_cart=u_hat,
                band_n_1based=BAND_N,
                labels=labels,
                onsite_csv=P1_ONSITE_CSV,
                fd_delta_e=P1_FD_DELTA_E,
                apply_and_reeval=P1_APPLY_AND_REEVAL,
                export_sens_csv=P1_EXPORT_SENS_CSV,
                export_summary_csv=P1_EXPORT_SUMMARY_CSV,
            )

        # ------------------ (B) Step-P2: Harrison + Slater–Koster λ(R,i,j) ------------------
        if P2_ENABLE:
            # Resolve reference wout path (needed for Wannier centers).
            hr_path_for_p2 = Path(HR_FILE)
            seed_for_p2 = _infer_seedname_from_hr(hr_path_for_p2)
            wout_ref_for_p2 = str(Path(WOUT_FILE)) if (WOUT_FILE and os.path.exists(WOUT_FILE)) else str(hr_path_for_p2.with_name(f"{seed_for_p2}.wout"))
            # Auto-fallback: if P2_POSCAR_DEF is missing, reuse P0 directory (if enabled).
            poscar_def_for_p2 = P2_POSCAR_DEF
            if (poscar_def_for_p2 is None) or (not os.path.exists(poscar_def_for_p2)):
                if P0_ENABLE and HR_FILE_P0 and os.path.exists(HR_FILE_P0):
                    _guess_poscar = str(Path(HR_FILE_P0).with_name("POSCAR"))
                    if os.path.exists(_guess_poscar):
                        poscar_def_for_p2 = _guess_poscar
            wout_def_for_p2 = P2_WOUT_DEF
            if wout_def_for_p2 is None:
                if P0_ENABLE and HR_FILE_P0 and os.path.exists(HR_FILE_P0):
                    _seed_p0 = _infer_seedname_from_hr(Path(HR_FILE_P0))
                    _guess_wout = str(Path(HR_FILE_P0).with_name(f"{_seed_p0}.wout"))
                    if os.path.exists(_guess_wout):
                        wout_def_for_p2 = _guess_wout
            run_P2_harrison_sk(
                H_R=H_R,
                R_list=R_list,
                degeneracy=degeneracy,
                lattice_ref=lattice,
                poscar_def=poscar_def_for_p2,
                wout_ref=wout_ref_for_p2,
                wout_def=wout_def_for_p2,
                k_frac=k_frac_used,
                u_hat_cart=u_hat,
                band_n_1based=BAND_N,
                labels=labels,
                knob_rows=knob_rows,
                use_sk=P2_USE_SK,
                eta_pp=P2_PP_PI_OVER_SIGMA,
                n_default=P2_EXPONENT_DEFAULT,
                n_dd=P2_EXPONENT_DD,
                min_abs_sk=P2_MIN_ABS_SK,
                apply_and_reeval=P2_APPLY_AND_REEVAL,
                export_lambda_csv=P2_EXPORT_LAMBDA_CSV,
                export_knob_csv=P2_EXPORT_KNOB_CSV,
            )

        # ------------------ Toy knob tuning: user-specified scalings ------------------
        if KNOB_TUNE_ENABLE and KNOB_TUNE_LIST:
            print("\n=== Toy knob tuning: apply user scalings and recompute curvature ===")

            H_R_tuned = apply_knob_scalings_Rij(
                H_R_in=H_R,
                R_list=R_list,
                scalings=KNOB_TUNE_LIST,
                one_based_orb=True,
            )
            Hk_t, D1_t, D2_t = build_H_D1_D2(w0, w1, w2, H_R_tuned)
            evals_t, evecs_t = np.linalg.eigh(Hk_t)
            band_t_1based, ov_t = band_track_index(evecs_ref=evecs0, evecs_new=evecs_t, idx_ref_1based=BAND_N)

            tuned = curvature_and_interband_table(
                Hk=Hk_t,
                D1=D1_t,
                D2=D2_t,
                band_n_1based=int(band_t_1based),
                band_m_1based=int(BAND_M)
            )
            print("\n[TOY] Curvature change summary:")
            print(f"  C0_total = {C0_total:+.6e} eV·Å^2")
            print(f"  C_tuned  = {tuned['C_total']:+.6e} eV·Å^2   (band overlap={ov_t:.6f}, chosen_band={band_t_1based})")
            print(f"  ΔC_total = {(tuned['C_total'] - C0_total):+.6e} eV·Å^2")

        # ------------------ Step C: P0 validation (true curvature from perturbed HR) ------------------
        if P0_ENABLE:
            print("\n=== P0 validation: compute true curvature from perturbed HR (Step C) ===")

            # True curvature on P0 Hamiltonian
            _, w0_p0, w1_p0, w2_p0, dotR_p0 = build_weights_and_dotR(
                lattice=lattice_p0,
                R_list=R_list_p0,
                degeneracy=degeneracy_p0,
                k_frac=k_frac_p0,
                u_hat=u_hat_p0,
            )
            Hk1, D1_1, D2_1 = build_H_D1_D2(w0_p0, w1_p0, w2_p0, H_R_p0)

            true1 = curvature_and_interband_table(
                Hk=Hk1,
                D1=D1_1,
                D2=D2_1,
                band_n_1based=int(band_n_p0_1based),
                band_m_1based=int(BAND_M)
            )
            C1_total = float(true1["C_total"])

            # Predicted ΔC from knob table (only if we had P0 HR loaded)
            pred_sum_intra = 0.0
            pred_sum_inter = 0.0
            pred_sum_total = 0.0
            if knob_rows and ("pred_dC_total" in knob_rows[0]):
                for rr in knob_rows:
                    try:
                        pred_sum_intra += float(rr.get("pred_dC_intra", 0.0))
                        pred_sum_inter += float(rr.get("pred_dC_inter", 0.0))
                        pred_sum_total += float(rr.get("pred_dC_total", 0.0))
                    except Exception:
                        pass

            # Real ΔC (note: depending on mode, k points may differ)
            dC_real = C1_total - float(C0_total)

            print(f"[P0] Baseline: k={k_frac_used}, band={BAND_N}, C0_total={float(C0_total):+.6e} eV·Å^2")
            print(f"[P0] Perturbed: k={list(map(float,k_frac_p0))}, band={band_n_p0_1based}, C1_total={C1_total:+.6e} eV·Å^2")
            print(f"[P0] ΔC_real = {dC_real:+.6e} eV·Å^2")

            if knob_rows and ("pred_dC_total" in knob_rows[0]):
                print(f"[P0] ΔC_pred(sum knobs, filtered) = {pred_sum_total:+.6e} eV·Å^2  (intra={pred_sum_intra:+.6e}, inter={pred_sum_inter:+.6e})")
                if abs(dC_real) > 0:
                    print(f"[P0] Coverage ΔC_pred/ΔC_real = {pred_sum_total/dC_real:+.4f}")
            else:
                print("[P0] No pred_dC_total column found (did not attach P0 HR into knob table).")

            # Optional geometry-only term (same H_R, but P0 lattice/direction)
            if P0_INCLUDE_GEOMETRY:
                _, w0_geo, w1_geo, w2_geo, dotR_geo = build_weights_and_dotR(
                    lattice=lattice_p0,
                    R_list=R_list,
                    degeneracy=degeneracy,
                    k_frac=k_frac_used,   # same fractional k
                    u_hat=u_hat_p0,
                )
                Hk_geo, D1_geo, D2_geo = build_H_D1_D2(w0_geo, w1_geo, w2_geo, H_R)

                # Track band by overlap (reference: baseline eigenvectors evecs0)
                evals_geo, evecs_geo = np.linalg.eigh(Hk_geo)
                band_geo_1based, ov_geo = band_track_index(evecs_ref=evecs0, evecs_new=evecs_geo, idx_ref_1based=BAND_N)

                geo = curvature_and_interband_table(
                    Hk=Hk_geo,
                    D1=D1_geo,
                    D2=D2_geo,
                    band_n_1based=int(band_geo_1based),
                    band_m_1based=int(BAND_M)
                )
                dC_geo = float(geo["C_total"]) - float(C0_total)
                print(f"[P0] Geometry-only (same HR, P0 lattice+dir): C_geo={float(geo['C_total']):+.6e}  ΔC_geo={dC_geo:+.6e}  (band overlap={ov_geo:.6f})")
# ------------------ Ablation loop (same as v3) ------------------
    ablation_rows = []
    for g in selected:
        idxs = group_to_indices.get(g, [])

        # If this group is absent (nR=0), ablation is a no-op and ΔC should be exactly zero.
        if len(idxs) == 0:
            row = {
                "group": str(g),
                "lambda": float(LAMBDA_ABLATE),
                "nR": 0,
                "chosen_band": int(BAND_N),
                "overlap": 1.0,
                "C_total": float(C0_total),
                "C_intra": float(C0_intra),
                "C_inter": float(C0_inter),
                "dC_total": 0.0,
                "dC_intra": 0.0,
                "dC_inter": 0.0,
            }
            ablation_rows.append(row)

            if EXPORT_FULL_INTERBAND_TABLE:
                # Write a copy of the baseline table so downstream scripts remain consistent.
                inter_g = base["inter_table"]
                inter_g_sorted = sorted(inter_g, key=lambda r: -abs(float(r["contrib"])))
                rows_g = []
                for rr in inter_g_sorted:
                    rows_g.append({"case": "ablation", "group": str(g), "lambda": float(LAMBDA_ABLATE), **rr})
                out_name = f"interband_contrib_group_{str(g)}_lambda{LAMBDA_ABLATE:.3f}.csv"
                write_csv(
                    out_name,
                    fieldnames=["case", "group", "lambda", "m", "Em", "dE", "V2", "contrib"],
                    rows=rows_g,
                )
            print(f"[INFO] group {str(g)} absent in HR (nR=0). Skip recompute; set ΔC=0.")
            continue

        scale = np.ones(nrpts, dtype=float)
        scale[idxs] *= float(LAMBDA_ABLATE)

        Hk_g, D1_g, D2_g = build_H_D1_D2(w0, w1, w2, H_R, scale_override=scale)
        evals_g, evecs_g = np.linalg.eigh(Hk_g)

        chosen_band, overlap = band_track_index(evecs_ref=evecs0, evecs_new=evecs_g, idx_ref_1based=BAND_N)

        res_g = curvature_and_interband_table(Hk_g, D1_g, D2_g, band_n_1based=chosen_band, band_m_1based=BAND_M)
        Cg_intra = float(res_g["C_intra"])
        Cg_inter = float(res_g["C_inter"])
        Cg_total = float(res_g["C_total"])

        row = {
            "group": str(g),
            "lambda": float(LAMBDA_ABLATE),
            "nR": int(len(idxs)),
            "chosen_band": int(chosen_band),
            "overlap": float(overlap),
            "C_total": Cg_total,
            "C_intra": Cg_intra,
            "C_inter": Cg_inter,
            "dC_total": Cg_total - C0_total,
            "dC_intra": Cg_intra - C0_intra,
            "dC_inter": Cg_inter - C0_inter,
        }
        ablation_rows.append(row)

        if EXPORT_FULL_INTERBAND_TABLE:
            inter_g = res_g["inter_table"]
            inter_g_sorted = sorted(inter_g, key=lambda r: -abs(float(r["contrib"])))
            rows_g = []
            for rr in inter_g_sorted:
                rows_g.append({"case": "ablation", "group": str(g), "lambda": float(LAMBDA_ABLATE), **rr})
            out_name = f"interband_contrib_group_{str(g)}_lambda{LAMBDA_ABLATE:.3f}.csv"
            write_csv(
                out_name,
                fieldnames=["case", "group", "lambda", "m", "Em", "dE", "V2", "contrib"],
                rows=rows_g,
            )

    ablation_sorted = sorted(ablation_rows, key=lambda r: -abs(float(r["dC_total"])))

    print("=== Ablation ranking by |ΔC_total| ===")
    for r in ablation_sorted:
        warn = ""
        if float(r["overlap"]) < MIN_OVERLAP_WARN:
            warn = "  [WARN: low overlap]"
        print(f"  group {r['group']:>6}  |ΔC|={abs(r['dC_total']):.6e}  ΔC={r['dC_total']:+.6e}  "
              f"ΔC_intra={r['dC_intra']:+.6e}  ΔC_inter={r['dC_inter']:+.6e}  "
              f"overlap={r['overlap']:.4f}  chosen_band={r['chosen_band']}  nR={r['nR']}{warn}")
    print("")

    write_csv(
        "ablation_ranking.csv",
        fieldnames=["group", "lambda", "nR", "chosen_band", "overlap",
                    "C_total", "C_intra", "C_inter",
                    "dC_total", "dC_intra", "dC_inter"],
        rows=ablation_sorted,
    )
    print("[OUT] Wrote ablation ranking to: ablation_ranking.csv")
    if EXPORT_FULL_INTERBAND_TABLE:
        print("[OUT] Wrote interband tables for each ablation group: interband_contrib_group_<group>_lambda*.csv")
    print("")
    print("[DONE]")

# =============================================================================
# Strain sweep driver (single-file mode)
# =============================================================================
# Usage:
#   (1) Single-case analysis (original behaviour):
#       python w90_harmonic_interband_ablation_v22.py
#   (2) Strain sweep (auto summary + plots):
#       python w90_harmonic_interband_ablation_v22.py --sweep
#
# Note: The sweep driver below is copied from w90_strain_sweep_driver_v5.py and
# embedded here so you only need ONE script file.
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w90_strain_sweep_driver_v3.py

Batch driver to run `w90_harmonic_interband_ablation_vXX.py` on a list of strain
folders and export:

  1) strain_summary.csv
  2) strain_vs_C.png                      (strain–C_total)
  3) strain_top_hopping_heatmap.png       (strain–top-hopping contributions)

Compared to v2:
  - Robust parsing for the new k-scan CSV schema written by the analysis script
    (kx_frac/ky_frac/kz_frac, band_index_tracked).
  - `strain_summary.csv` now contains structured top-hopping columns:
        top1_label, top1_R1, top1_R2, top1_R3, top1_dC, top1_lambda, top1_dlam, ...
    (source is selectable: P0 / P2 / auto).
  - Fixes the ValueError: "cannot convert float NaN to integer" by allowing
    missing tracked-band columns.

Usage:
  1) Edit the USER PARAMETERS block.
  2) Run:
        python w90_strain_sweep_driver_v3.py

Folder layout (example):
  BASE_DIR/
    0%/wannier90_hr.dat  POSCAR  wannier90.win  wannier90.wout  wannier90_band.dat
    2%/wannier90_hr.dat  POSCAR  wannier90.win  wannier90.wout  wannier90_band.dat
    ...

Notes:
  - The driver runs ONE full analysis per strain folder to auto-pick k* (min |v|)
    on your chosen k-line.
  - Then it runs a "reference@k*" job in a temporary directory to compute:
        C_ref(k*), ΔC_real, and the knob predictions (P0/P2).
"""


import csv
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib.util


# =========================
# USER PARAMETERS (EDIT ME)
# =========================

BASE_DIR = Path(".")

# Strain folders to sweep (folder names; they will also be used as labels)
STRAIN_DIRS = [
    "0%",
    "2%",
]

REF_DIR = "0%"  # reference folder

# Filenames inside each folder
# Preferred: set SWEEP_SEEDNAME and keep *_NAME as None for auto-inference.
SWEEP_SEEDNAME = "wannier90"  # -> {seed}_hr.dat / {seed}.win / {seed}.wout
HR_NAME: Optional[str] = None
POSCAR_NAME = "POSCAR"
WIN_NAME: Optional[str] = None
WOUT_NAME: Optional[str] = None

# Analysis script to drive
ANALYSIS_SCRIPT = Path(__file__).resolve()  # this script (single-file mode)  # set to your latest analysis script

# Band / direction config
NUM_WANN = 28
BAND_N = 17
BAND_M = 16

DIR_MODE = "from_kline"
KLINE_START = [0.0, 0.0, 0.0]
KLINE_END = [0.0, 0.5, 0.0]

# Auto-k* selection (min |v| for band BAND_N)
AUTO_K0_ENABLE = True
AUTO_K0_NPTS = 101
AUTO_K0_REFINE_ROOT = True
AUTO_K0_OVERLAP_TRACK = True

# Predictions
ENABLE_P0_PREDICTION = True   # HR-ratio (reference -> strain)
ENABLE_P2_PREDICTION = True   # Harrison + Slater–Koster (geometry-only)
# P0 k-point mode used inside the analysis script:
#   - "same_k"       : compare all strains at the SAME k* (the k* found in each case run)
#   - "each_minabs_v": for P0, re-find each structure's own min-|v| k* and compare at those points
# For a clean strain→curvature trend, "same_k" is recommended.
P0_KPOINT_MODE = "same_k"


# Summary CSV + structured top hopping
SUMMARY_CSV = BASE_DIR / "strain_summary.csv"
TOP_HOPPING_N = 6  # keep top-6 knobs in structured columns

# Which prediction to use for the structured top{n}_* columns and for the heatmap:
#   "auto" : prefer P0 if enabled, else P2
#   "P0"   : use P0 only
#   "P2"   : use P2 only
TOP_SOURCE = "auto"

# Plot outputs
EXPORT_PLOTS = True
PLOT_C_FILE = BASE_DIR / "strain_vs_C.png"
PLOT_HEATMAP_FILE = BASE_DIR / "strain_top_hopping_heatmap.png"
HEATMAP_TOPK_GLOBAL = 12  # number of global knobs to show in heatmap

# Temporary directory
KEEP_TMP_OUTPUTS = False
TMP_ROOT = BASE_DIR / "_strain_sweep_tmp"

# For derivatives: "win" recommended if your win has unit_cell_cart in Angstrom.
LATTICE_SOURCE = "win"  # "win" or "poscar"

# Backward-compatible aliases
DO_P0_PRED = ENABLE_P0_PREDICTION
DO_P2_PRED = ENABLE_P2_PREDICTION


def _resolve_sweep_file_names() -> Tuple[str, str, str, str]:
    """Resolve sweep filenames from explicit overrides or seedname defaults."""
    seed = str(SWEEP_SEEDNAME).strip() or "wannier90"
    hr_name = str(HR_NAME).strip() if HR_NAME else f"{seed}_hr.dat"
    win_name = str(WIN_NAME).strip() if WIN_NAME else f"{seed}.win"
    wout_name = str(WOUT_NAME).strip() if WOUT_NAME else f"{seed}.wout"
    poscar_name = str(POSCAR_NAME).strip() or "POSCAR"
    return hr_name, poscar_name, win_name, wout_name

# =========================
# Internal helpers
# =========================


@dataclass
class KStarInfo:
    kx_frac: float
    ky_frac: float
    kz_frac: float
    t: float
    E_eV: float
    v_eVAng: float
    abs_v_eVAng: float
    tracked_band: int
    overlap: float


def _load_analysis_module(script_path: Path):
    """Load a *fresh* analysis module object (avoid global-state carry-over)."""
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Analysis script not found: {script_path}")
    mod_name = f"w90_ablation_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import analysis script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _run_analysis(mod, workdir: Path, overrides: Dict[str, Any]) -> None:
    """Run mod.main() with selected global overrides in a given workdir."""
    compatibility_map = {
        # legacy/driver aliases -> canonical names used in main logic
        "TOPN": "TOPN_BANDS",
        "KNOB_GROUP_MAX": "KNOB_MAX_GROUP",
        "P0_POSCAR_FILE": "POSCAR_P0",
        "P0_HR_FILE": "HR_FILE_P0",
    }

    workdir.mkdir(parents=True, exist_ok=True)
    for k, v in overrides.items():
        canonical_key = compatibility_map.get(k, k)
        if not hasattr(mod, canonical_key):
            print(f"[WARN] Unknown override key skipped: {k} (canonical: {canonical_key})")
            continue
        setattr(mod, canonical_key, v)

    if hasattr(mod, "_normalize_runtime_params"):
        mod._normalize_runtime_params()

    cwd = Path.cwd()
    try:
        os.chdir(workdir)
        mod.main()
    finally:
        os.chdir(cwd)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _to_int(x: Any, default: int = -1) -> int:
    """Safe int parser: NaN/None -> default."""
    v = _to_float(x, default=float("nan"))
    if not (v == v):
        return default
    return int(round(v))


def _first_nonempty(row: Dict[str, str], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in row:
            v = str(row.get(k, "")).strip()
            if v != "":
                return v
    return None


def _parse_kstar(scan_csv: Path) -> KStarInfo:
    """Parse kline_scan_minabs_v.csv produced by the analysis script.

    The analysis v21+ writes columns:
      kx_frac, ky_frac, kz_frac, band_index_tracked, overlap,
      E_eV, v_eVAng, abs_v_eVAng, (plus others)

    This parser is backward-compatible with older schemas (kx/ky/kz, tracked_band).
    """
    if not scan_csv.exists():
        raise FileNotFoundError(f"Missing k-scan table: {scan_csv}")
    rows = _read_csv_rows(scan_csv)
    if not rows:
        raise RuntimeError(f"Empty k-scan table: {scan_csv}")

    def abs_v(row: Dict[str, str]) -> float:
        return abs(_to_float(row.get("abs_v_eVAng", "nan")))

    best = min(rows, key=abs_v)

    kx = _to_float(_first_nonempty(best, ["kx_frac", "kx", "k_frac_x"]))
    ky = _to_float(_first_nonempty(best, ["ky_frac", "ky", "k_frac_y"]))
    kz = _to_float(_first_nonempty(best, ["kz_frac", "kz", "k_frac_z"]))
    t = _to_float(_first_nonempty(best, ["t", "t_frac", "line_t"]))

    tracked = _to_int(_first_nonempty(best, ["band_index_tracked", "tracked_band", "tracked_band_index", "tracked_idx"]))
    overlap = _to_float(_first_nonempty(best, ["overlap", "overlap_track", "overlap_to_prev"]))

    return KStarInfo(
        kx_frac=kx,
        ky_frac=ky,
        kz_frac=kz,
        t=t,
        E_eV=_to_float(best.get("E_eV")),
        v_eVAng=_to_float(best.get("v_eVAng")),
        abs_v_eVAng=_to_float(best.get("abs_v_eVAng")),
        tracked_band=tracked,
        overlap=overlap,
    )


def _parse_C_total(hr_num_csv: Path) -> float:
    if not hr_num_csv.exists():
        raise FileNotFoundError(f"Missing HR numerical check: {hr_num_csv}")
    rows = _read_csv_rows(hr_num_csv)
    if not rows:
        raise RuntimeError(f"Empty HR numerical check: {hr_num_csv}")
    return _to_float(rows[0].get("C_analytic"))


def _parse_strain_value(label: str) -> float:
    """Parse numeric strain from folder label like '2%' or '-1.5%'."""
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", label)
    if not m:
        return float("nan")
    return float(m.group(1))


def _parse_knob_table(
    knob_csv: Path,
    dC_col: str,
    topn: int,
    source_tag: str,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, float], str]:
    """Parse a knob sensitivity table.

    Returns:
      sum_dC: float
      top_entries: list(dict)
      contrib_map: dict(key -> dC)
      keywords: human-readable string
    """
    if not knob_csv.exists():
        return (float("nan"), [], {}, "")
    rows = _read_csv_rows(knob_csv)
    if not rows:
        return (float("nan"), [], {}, "")

    def row_key(r: Dict[str, str]) -> str:
        R1 = r.get("R1", "?")
        R2 = r.get("R2", "?")
        R3 = r.get("R3", "?")
        li = r.get("label_i", r.get("i", "?"))
        lj = r.get("label_j", r.get("j", "?"))
        return f"R=({R1},{R2},{R3}) {li}-{lj}"

    # contributions
    contrib: Dict[str, float] = {}
    dcs: List[float] = []
    for r in rows:
        dC = _to_float(r.get(dC_col))
        if not (dC == dC):
            continue
        dcs.append(dC)
        contrib[row_key(r)] = dC
    sum_dC = float(sum(dcs)) if dcs else float("nan")

    def score(r: Dict[str, str]) -> float:
        return abs(_to_float(r.get(dC_col)))

    top = sorted(rows, key=score, reverse=True)[: max(0, int(topn))]

    top_entries: List[Dict[str, Any]] = []
    kw_list: List[str] = []
    for r in top:
        dC = _to_float(r.get(dC_col))
        if not (dC == dC):
            continue
        R1 = _to_int(r.get("R1"), default=0)
        R2 = _to_int(r.get("R2"), default=0)
        R3 = _to_int(r.get("R3"), default=0)
        li = r.get("label_i", r.get("i", "?"))
        lj = r.get("label_j", r.get("j", "?"))
        # lambda/dlambda columns differ between P0 and P2
        lam = _to_float(_first_nonempty(r, ["lambda_P2", "lambda_fit", "lambda"]))
        dlam = _to_float(_first_nonempty(r, ["dlam_P2", "delta_lambda", "dlam"]))

        key = row_key(r)
        top_entries.append({
            "key": key,
            "label": f"{li}-{lj}",
            "label_i": li,
            "label_j": lj,
            "R1": R1,
            "R2": R2,
            "R3": R3,
            "dC": dC,
            "lambda": lam,
            "dlam": dlam,
            "source": source_tag,
        })

        if lam == lam and dlam == dlam:
            kw = f"R=({R1},{R2},{R3}) {li}-{lj} dC={dC:+.3e} (λ={lam:.6f}, Δλ={dlam:+.6f})"
        else:
            kw = f"R=({R1},{R2},{R3}) {li}-{lj} dC={dC:+.3e}"
        kw_list.append(kw)

    return (sum_dC, top_entries, contrib, " | ".join(kw_list))


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write in summary CSV.")
    header: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in header:
                header.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _ratio(a: float, b: float) -> float:
    if not (a == a) or not (b == b):
        return float("nan")
    if abs(b) < 1e-14:
        return float("nan")
    return a / b


def _maybe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _export_plots(
    summary_rows: List[Dict[str, Any]],
    heatmap_data: Dict[str, Dict[str, float]],
    out_c: Path,
    out_heat: Path,
    heat_topk: int,
) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        print("[WARN] matplotlib not available; skip plotting.")
        return

    # ---- (1) strain vs C_total ----
    strains: List[float] = []
    Cvals: List[float] = []
    for r in summary_rows:
        strains.append(_to_float(r.get("strain_value", "nan")))
        Cvals.append(_to_float(r.get("C_strain_eVAng2", "nan")))
    # sort by strain
    order = sorted(range(len(strains)), key=lambda i: (float("inf") if not (strains[i] == strains[i]) else strains[i]))
    strains_s = [strains[i] for i in order]
    Cvals_s = [Cvals[i] for i in order]

    plt.figure()
    plt.plot(strains_s, Cvals_s, marker="o")
    plt.xlabel("strain (%)")
    plt.ylabel("C_total (eV·Å²)")
    plt.tight_layout()
    plt.savefig(out_c, dpi=200)
    plt.close()
    print(f"[OUT] Wrote plot: {out_c}")

    # ---- (2) heatmap: strain vs top-hopping dC ----
    # Build global knob list by max |dC| across strains
    knob_scores: Dict[str, float] = {}
    for sdir, mp in heatmap_data.items():
        for k, v in mp.items():
            knob_scores[k] = max(knob_scores.get(k, 0.0), abs(float(v)))
    knobs_sorted = sorted(knob_scores.keys(), key=lambda k: knob_scores[k], reverse=True)
    knobs = knobs_sorted[: max(1, int(heat_topk))] if knobs_sorted else []
    if not knobs:
        print("[WARN] No knob data for heatmap; skip heatmap plot.")
        return

    # rows: strains (in the same sorted order)
    sdirs_order = [summary_rows[i]["strain_dir"] for i in order]
    mat: List[List[float]] = []
    for sdir in sdirs_order:
        mp = heatmap_data.get(str(sdir), {})
        mat.append([float(mp.get(k, 0.0)) for k in knobs])

    import numpy as np
    A = np.array(mat, dtype=float)

    plt.figure(figsize=(max(6, 0.5 * len(knobs)), max(3, 0.35 * len(sdirs_order))))
    im = plt.imshow(A, aspect="auto", origin="lower")
    plt.colorbar(im, label="dC (eV·Å²)")
    plt.yticks(range(len(sdirs_order)), sdirs_order)
    plt.xticks(range(len(knobs)), knobs, rotation=90)
    plt.ylabel("strain")
    plt.xlabel("top hopping knobs")
    plt.tight_layout()
    plt.savefig(out_heat, dpi=200)
    plt.close()
    print(f"[OUT] Wrote plot: {out_heat}")


def strain_sweep_main() -> None:
    _normalize_runtime_params()
    hr_name, poscar_name, win_name, wout_name = _resolve_sweep_file_names()
    base = BASE_DIR.resolve()
    ref_dir = (base / REF_DIR).resolve()
    ref_hr = (ref_dir / hr_name).resolve()
    ref_poscar = (ref_dir / poscar_name).resolve()
    ref_win = (ref_dir / win_name).resolve()
    ref_wout = (ref_dir / wout_name).resolve()

    if not ref_hr.exists():
        raise FileNotFoundError(f"Reference HR missing: {ref_hr}")

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []

    # heatmap_data[strain_dir] = { knob_key : dC_pred }
    heatmap_data: Dict[str, Dict[str, float]] = {}

    def _common_overrides(*, hr: Path, poscar: Path, win: Path, wout: Path,
                          lattice_source: str) -> Dict[str, Any]:
        return dict(
            HR_FILE=str(hr),
            POSCAR_FILE=str(poscar) if poscar.exists() else "",
            WIN_FILE=str(win) if win.exists() else "",
            WOUT_FILE=str(wout) if wout.exists() else "",
            NUM_WANN=NUM_WANN,
            BAND_N=BAND_N,
            BAND_M=BAND_M,
            TOPN_BANDS=8,
            DIR_MODE=DIR_MODE,
            KLINE_START=KLINE_START,
            KLINE_END=KLINE_END,
            LATTICE_SOURCE=lattice_source,
        )

    for sdir in STRAIN_DIRS:
        case_dir = (base / sdir).resolve()
        case_hr = (case_dir / hr_name).resolve()
        case_poscar = (case_dir / poscar_name).resolve()
        case_win = (case_dir / win_name).resolve()
        case_wout = (case_dir / wout_name).resolve()

        if not case_hr.exists():
            raise FileNotFoundError(f"[{sdir}] HR missing: {case_hr}")

        # ----------------------------
        # (1) Run strain analysis to get k* + C_strain
        # ----------------------------
        mod_case = _load_analysis_module(ANALYSIS_SCRIPT)

        win_used = case_win if case_win.exists() else ref_win
        wout_used = case_wout if case_wout.exists() else ref_wout

        lattice_source_case = LATTICE_SOURCE
        if str(LATTICE_SOURCE).lower() == "win" and (not win_used.exists()):
            lattice_source_case = "poscar"

        case_overrides = _common_overrides(
            hr=case_hr,
            poscar=(case_poscar if case_poscar.exists() else ref_poscar),
            win=win_used,
            wout=wout_used,
            lattice_source=lattice_source_case,
        )
        case_overrides.update(dict(
            # auto k*
            AUTO_K0_MINABS_V_ENABLE=AUTO_K0_ENABLE,
            AUTO_K0_MINABS_V_NPTS=AUTO_K0_NPTS,
            AUTO_K0_REFINE_ROOT=AUTO_K0_REFINE_ROOT,
            AUTO_K0_OVERLAP_TRACK=AUTO_K0_OVERLAP_TRACK,
            EXPORT_KSCAN_TABLE=True,
            # speed: disable heavy exports
            EXPORT_FULL_INTERBAND_TABLE=False,
            EXPORT_ORBPAIR_GROUP_RANKING=False,
            EXPORT_ORBPAIR_BY_R_TOP=False,
            ABLATE_GROUP_MODE="list",
            ABLATE_GROUP_LIST=[],
            KNOB_SENS_ENABLE=False,
            P0_ENABLE=False,
            P2_ENABLE=False,
            BAND_CHECK_ENABLE=False,
            HR_NUM_CHECK_ENABLE=True,
            EXPORT_HR_NUM_CHECK=True,
        ))

        print(f"\n===== [{sdir}] Running strain analysis (auto k*) =====")
        _run_analysis(mod_case, case_dir, case_overrides)

        kstar = _parse_kstar(case_dir / "kline_scan_minabs_v.csv")
        C_case = _parse_C_total(case_dir / "hr_numerical_check.csv")

        # ----------------------------
        # (2) Reference evaluation at same k* + predictions
        # ----------------------------
        tmp_case = TMP_ROOT / f"ref_at_kstar__{sdir.replace('%', 'pct')}"
        if tmp_case.exists():
            shutil.rmtree(tmp_case)
        tmp_case.mkdir(parents=True, exist_ok=True)

        mod_ref = _load_analysis_module(ANALYSIS_SCRIPT)

        lattice_source_ref = LATTICE_SOURCE
        if str(LATTICE_SOURCE).lower() == "win" and (not ref_win.exists()):
            lattice_source_ref = "poscar"

        ref_overrides = _common_overrides(
            hr=ref_hr,
            poscar=ref_poscar,
            win=ref_win,
            wout=ref_wout,
            lattice_source=lattice_source_ref,
        )
        ref_overrides.update(dict(
            # manual k*
            AUTO_K0_MINABS_V_ENABLE=False,
            K_FRAC=[kstar.kx_frac, kstar.ky_frac, kstar.kz_frac],
            EXPORT_KSCAN_TABLE=False,
            # speed: no ablation/orbpair
            EXPORT_FULL_INTERBAND_TABLE=False,
            EXPORT_ORBPAIR_GROUP_RANKING=False,
            EXPORT_ORBPAIR_BY_R_TOP=False,
            ABLATE_GROUP_MODE="list",
            ABLATE_GROUP_LIST=[],
            BAND_CHECK_ENABLE=False,
            HR_NUM_CHECK_ENABLE=True,
            EXPORT_HR_NUM_CHECK=True,
            # knob sensitivities
            KNOB_SENS_ENABLE=True,
            KNOB_MAX_GROUP=5,
        ))

        # P0: attach strained HR into the knob table
        if ENABLE_P0_PREDICTION:
            # IMPORTANT: analysis script (v21+) uses HR_FILE_P0 / POSCAR_P0 internally.
            # Earlier driver drafts used P0_HR_FILE / P0_POSCAR_FILE. We set BOTH.
            ref_overrides.update(
                dict(
                    P0_ENABLE=True,
                    P0_KPOINT_MODE=P0_KPOINT_MODE,
                    # v21 canonical names
                    HR_FILE_P0=str(case_hr),
                    POSCAR_P0=str(case_poscar) if case_poscar.exists() else "",
                    WIN_FILE_P0=str(case_win) if case_win.exists() else "",
                    WOUT_FILE_P0=str(case_wout) if case_wout.exists() else "",
                )
            )
        else:
            ref_overrides.update(dict(P0_ENABLE=False))

        # P2: geometry-only mapping
        if ENABLE_P2_PREDICTION:
            ref_overrides.update(
                dict(
                    P2_ENABLE=True,
                    P2_POSCAR_DEF=str(case_poscar) if case_poscar.exists() else "",
                    P2_WOUT_DEF=str(case_wout) if case_wout.exists() else "",
                    P2_EXPORT_LAMBDA_CSV="p2_lambda_map.csv",
                    P2_APPLY_AND_REEVAL=False,
                )
            )
        else:
            ref_overrides.update(dict(P2_ENABLE=False))

        print(f"\n===== [{sdir}] Running reference @ k* + predictions =====")
        _run_analysis(mod_ref, tmp_case, ref_overrides)

        C_ref = _parse_C_total(tmp_case / "hr_numerical_check.csv")
        dC_real = C_case - C_ref

        # ---- parse P0/P2 knob tables ----
        dC_pred_p0, top_p0, map_p0, kw_p0 = (float("nan"), [], {}, "")
        if ENABLE_P0_PREDICTION:
            dC_pred_p0, top_p0, map_p0, kw_p0 = _parse_knob_table(
                tmp_case / "knob_sensitivity.csv",
                dC_col="pred_dC_total",
                topn=TOP_HOPPING_N,
                source_tag="P0",
            )

        dC_pred_p2, top_p2, map_p2, kw_p2 = (float("nan"), [], {}, "")
        if ENABLE_P2_PREDICTION:
            dC_pred_p2, top_p2, map_p2, kw_p2 = _parse_knob_table(
                tmp_case / "knob_sensitivity_with_p2.csv",
                dC_col="dC_pred_P2",
                topn=TOP_HOPPING_N,
                source_tag="P2",
            )

        # choose structured/heatmap source
        src = TOP_SOURCE.lower().strip()
        use_source: str
        if src == "p0":
            use_source = "P0"
        elif src == "p2":
            use_source = "P2"
        else:
            use_source = "P0" if ENABLE_P0_PREDICTION else ("P2" if ENABLE_P2_PREDICTION else "")

        top_used = top_p0 if use_source == "P0" else top_p2
        map_used = map_p0 if use_source == "P0" else map_p2
        heatmap_data[str(sdir)] = dict(map_used)

        # ---- summary row ----
        row: Dict[str, Any] = {
            "strain_dir": sdir,
            "strain_value": f"{_parse_strain_value(sdir):.6f}",
            "k*_frac_x": f"{kstar.kx_frac:.10f}",
            "k*_frac_y": f"{kstar.ky_frac:.10f}",
            "k*_frac_z": f"{kstar.kz_frac:.10f}",
            "t_on_kline": f"{kstar.t:.10f}",
            "E_strain_eV": f"{kstar.E_eV:.10f}",
            "v_strain_eVAng": f"{kstar.v_eVAng:.6e}",
            "abs_v_strain_eVAng": f"{kstar.abs_v_eVAng:.6e}",
            "tracked_band": kstar.tracked_band,
            "overlap_track": f"{kstar.overlap:.6f}",
            "C_strain_eVAng2": f"{C_case:.10e}",
            "C_ref_eVAng2": f"{C_ref:.10e}",
            "dC_real_eVAng2": f"{dC_real:.10e}",
            "dC_pred_P0_eVAng2": f"{dC_pred_p0:.10e}" if ENABLE_P0_PREDICTION else "",
            "dC_pred_P2_eVAng2": f"{dC_pred_p2:.10e}" if ENABLE_P2_PREDICTION else "",
            "dC_real_over_pred_P0": f"{_ratio(dC_real, dC_pred_p0):.6f}" if ENABLE_P0_PREDICTION else "",
            "dC_real_over_pred_P2": f"{_ratio(dC_real, dC_pred_p2):.6f}" if ENABLE_P2_PREDICTION else "",
            "top_hoppings_P0": kw_p0,
            "top_hoppings_P2": kw_p2,
            "top_source_used": use_source,
        }

        # Structured top columns (top1_label, top1_R1, ..., top1_dC, ...)
        for idx in range(1, TOP_HOPPING_N + 1):
            entry = top_used[idx - 1] if idx - 1 < len(top_used) else None
            row[f"top{idx}_label"] = entry["label"] if entry else ""
            row[f"top{idx}_R"] = f"({entry['R1']},{entry['R2']},{entry['R3']})" if entry else ""
            row[f"top{idx}_R1"] = entry["R1"] if entry else ""
            row[f"top{idx}_R2"] = entry["R2"] if entry else ""
            row[f"top{idx}_R3"] = entry["R3"] if entry else ""
            row[f"top{idx}_dC"] = f"{entry['dC']:+.6e}" if entry else ""
            lam = entry.get("lambda") if entry else float("nan")
            dlam = entry.get("dlam") if entry else float("nan")
            row[f"top{idx}_lambda"] = f"{lam:.8f}" if (lam == lam) else ""
            row[f"top{idx}_dlam"] = f"{dlam:+.8f}" if (dlam == dlam) else ""

        summary_rows.append(row)

        if not KEEP_TMP_OUTPUTS:
            shutil.rmtree(tmp_case, ignore_errors=True)

    _write_summary_csv(SUMMARY_CSV, summary_rows)
    print(f"\n[OK] Wrote summary to: {SUMMARY_CSV.resolve()}")

    if EXPORT_PLOTS:
        _export_plots(
            summary_rows=summary_rows,
            heatmap_data=heatmap_data,
            out_c=PLOT_C_FILE,
            out_heat=PLOT_HEATMAP_FILE,
            heat_topk=HEATMAP_TOPK_GLOBAL,
        )



def _entrypoint():
    import sys
    if "--sweep" in sys.argv:
        strain_sweep_main()
    else:
        main()


if __name__ == "__main__":
    _entrypoint()
