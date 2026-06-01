#!/usr/bin/env python3
"""
課題1: 2次元強磁性イジング模型の転移温度
古典モンテカルロ法（メトロポリスアルゴリズム）による数値計算

実行: python ising_mc.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 定数
# ============================================================
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))  # Onsager厳密解 ≈ 2.26919


# ============================================================
# コアシミュレーション関数
# ============================================================

def init_spins(L, mode="random", rng=None):
    """L×L スピン格子を初期化する。

    mode='random'  : ランダム配置（高温向き）
    mode='ordered' : 全スピン +1 整列（低温向き）
    """
    if rng is None:
        rng = np.random.default_rng()
    if mode == "random":
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    else:
        return np.ones((L, L), dtype=np.int8)


def compute_energy(spins, J=1.0):
    """全エネルギーを計算する（周期境界条件）。

    H = -J Σ_{<i,j>} s_i s_j
    正方格子の各結合を1回だけ数えるため roll(+1) のみ使う。
    """
    E = -J * float(np.sum(
        spins * np.roll(spins, 1, axis=0) +
        spins * np.roll(spins, 1, axis=1)
    ))
    return E


def metropolis_sweep(spins, beta, J=1.0, rng=None):
    """チェッカーボード分解による1 MCS（メトロポリス）。

    格子を白・黒の2副格子に分け、各副格子内でサイトを同時に更新する。
    同じ副格子のサイトは互いに最近接でないため、独立に更新できる。
    """
    if rng is None:
        rng = np.random.default_rng()
    L = spins.shape[0]
    ii, jj = np.mgrid[0:L, 0:L]

    for parity in (0, 1):
        # 副格子マスク（チェッカーボードの白/黒）
        mask = (ii + jj) % 2 == parity

        # 最近接スピンの和（周期境界条件）
        neighbor_sum = (
            np.roll(spins,  1, axis=0) +
            np.roll(spins, -1, axis=0) +
            np.roll(spins,  1, axis=1) +
            np.roll(spins, -1, axis=1)
        )

        # ΔE = 2J s_i Σ_{δ} s_{i+δ}
        dE = 2.0 * J * spins * neighbor_sum

        # メトロポリス受理確率: min(1, exp(-β ΔE))
        rand = rng.random(size=(L, L))
        accept = (dE <= 0) | (rand < np.exp(-beta * dE))

        # この副格子の受理サイトのみ反転
        spins[mask & accept] *= -1


def run_simulation(L, T, N_therm, N_meas,
                   init_mode="random", J=1.0, seed=None):
    """モンテカルロシミュレーションを実行し、熱平均物理量を返す。

    Returns
    -------
    dict:
        avg_absm : <|m|>
        chi      : 磁化率 χ = L^2/T (<m^2> - <|m|>^2)
        U_L      : Binderキュムラント 1 - <m^4>/(3<m^2>^2)
        avg_m2   : <m^2>
        avg_m4   : <m^4>
    """
    rng = np.random.default_rng(seed)
    spins = init_spins(L, mode=init_mode, rng=rng)
    beta = 1.0 / T

    # 熱化フェーズ（物理量は記録しない）
    for _ in range(N_therm):
        metropolis_sweep(spins, beta, J, rng)

    # 測定フェーズ
    m2_arr   = np.empty(N_meas)
    absm_arr = np.empty(N_meas)
    m4_arr   = np.empty(N_meas)

    for n in range(N_meas):
        metropolis_sweep(spins, beta, J, rng)
        m = float(spins.mean())
        m2_arr[n]   = m * m
        absm_arr[n] = abs(m)
        m4_arr[n]   = m * m * m * m

    avg_m2   = m2_arr.mean()
    avg_absm = absm_arr.mean()
    avg_m4   = m4_arr.mean()

    chi = L**2 / T * (avg_m2 - avg_absm**2)

    # <m^2>^2 がほぼゼロなら（極端な高温等）U_L = 0 とする
    denom = 3.0 * avg_m2**2
    U_L = 0.0 if denom < 1e-30 else 1.0 - avg_m4 / denom

    return {
        "avg_absm": avg_absm,
        "chi":      chi,
        "U_L":      U_L,
        "avg_m2":   avg_m2,
        "avg_m4":   avg_m4,
    }


# ============================================================
# ユニットテスト
# ============================================================

def test_energy():
    """全スピン整列配置で H = -2J L^2 となることを確認（正方格子・周期境界）。"""
    for L in (2, 4, 8):
        spins = np.ones((L, L), dtype=np.int8)
        E = compute_energy(spins, J=1.0)
        expected = -2.0 * L**2  # 各サイト4結合、重複除去で2結合/サイト
        assert abs(E - expected) < 1e-10, f"L={L}: E={E}, expected={expected}"
    print("PASS  test_energy")


def test_high_temperature():
    """十分高温（T=100）では磁化がゼロ付近に収束することを確認。"""
    L, T = 16, 100.0
    rng = np.random.default_rng(0)
    spins = init_spins(L, mode="ordered", rng=rng)
    beta = 1.0 / T
    for _ in range(2000):
        metropolis_sweep(spins, beta, rng=rng)
    m = abs(float(spins.mean()))
    assert m < 0.2, f"High T: |m|={m:.3f}, expected ≈ 0"
    print(f"PASS  test_high_temperature  (|m|={m:.3f})")


def test_low_temperature():
    """十分低温（T=0.1）では磁化が ±1 付近に収束することを確認。"""
    L, T = 8, 0.1
    rng = np.random.default_rng(0)
    spins = init_spins(L, mode="random", rng=rng)
    beta = 1.0 / T
    for _ in range(2000):
        metropolis_sweep(spins, beta, rng=rng)
    m = abs(float(spins.mean()))
    assert m > 0.95, f"Low T: |m|={m:.3f}, expected ≈ 1"
    print(f"PASS  test_low_temperature   (|m|={m:.3f})")


def run_unit_tests():
    print("=" * 50)
    print("ユニットテスト")
    print("=" * 50)
    test_energy()
    test_high_temperature()
    test_low_temperature()
    print()


# ============================================================
# O(L^2) スケーリングテスト
# ============================================================

def scaling_test(outdir="."):
    """1 MCS の計算時間が O(L^2) でスケールすることを確認する。"""
    print("=" * 50)
    print("O(L^2) スケーリングテスト")
    print("=" * 50)

    L_list  = [8, 16, 32, 64, 128, 256, 512, 1024]
    N_rep   = 20   # 各 L で繰り返す MCS 数（平均を取る）
    beta    = 1.0 / 2.3
    rng     = np.random.default_rng(0)
    times   = []

    for L in L_list:
        spins = init_spins(L, rng=rng)
        # ウォームアップ（JIT等の効果を排除）
        for _ in range(3):
            metropolis_sweep(spins, beta, rng=rng)
        # 計測
        t0 = time.perf_counter()
        for _ in range(N_rep):
            metropolis_sweep(spins, beta, rng=rng)
        t1 = time.perf_counter()
        t_ms = (t1 - t0) / N_rep * 1e3  # ms/MCS
        times.append(t_ms)
        print(f"  L={L:4d}:  {t_ms:.3f} ms/MCS")

    L_arr = np.array(L_list, dtype=float)
    t_arr = np.array(times)
    slope = np.polyfit(np.log(L_arr), np.log(t_arr), 1)[0]
    print(f"\n  log-log 傾き = {slope:.3f}  (理論値 ≈ 2.0)\n")

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.loglog(L_arr, t_arr, "o-", color="steelblue", label="計測値")
    ref = t_arr[0] * (L_arr / L_arr[0])**2
    ax1.loglog(L_arr, ref, "--", color="gray", label="$O(L^2)$ 参照線")
    ax1.set_xlabel("$L$")
    ax1.set_ylabel("計算時間 [ms / MCS]")
    ax1.set_title("$O(L^2)$ スケーリング（log-log）")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.text(0.05, 0.95, f"傾き = {slope:.2f}", transform=ax1.transAxes,
             va="top", fontsize=10)

    ax2.plot(L_arr, t_arr / L_arr**2 * 1e6, "o-", color="darkorange")
    ax2.set_xlabel("$L$")
    ax2.set_ylabel("$t / L^2$ [μs]")
    ax2.set_title("$t / L^2$ vs $L$（一定に収束すれば $O(L^2)$）")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{outdir}/scaling_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  → {path} 保存\n")


# ============================================================
# メイン計算
# ============================================================

def main():
    outdir = "."   # 出力先（このスクリプトと同じフォルダ）

    run_unit_tests()

    # ---- シミュレーションパラメータ ----
    L_list  = [8, 16, 32, 64, 128]
    T_list  = np.linspace(1.8, 2.8, 35)
    N_therm = 1000
    N_meas  = 5000

    print("=" * 50)
    print("メイン計算")
    print("=" * 50)
    print(f"L        = {L_list}")
    print(f"T 範囲   = [{T_list[0]:.2f}, {T_list[-1]:.2f}]  ({len(T_list)} 点)")
    print(f"N_therm  = {N_therm}")
    print(f"N_meas   = {N_meas}")
    print(f"T_c（厳密解） = {TC_EXACT:.6f}")
    print()

    # 結果格納
    avg_absm = {L: [] for L in L_list}
    chi      = {L: [] for L in L_list}
    U_L      = {L: [] for L in L_list}

    t_start = time.time()

    for i_L, L in enumerate(L_list):
        for i_T, T in enumerate(T_list):
            seed = 42 + i_L * 1000 + i_T
            res = run_simulation(L, T, N_therm, N_meas,
                                 init_mode="random", seed=seed)
            avg_absm[L].append(res["avg_absm"])
            chi[L].append(res["chi"])
            U_L[L].append(res["U_L"])

        elapsed = time.time() - t_start
        print(f"  L={L:3d} 完了  ({elapsed:.1f} s 経過)")

    print()

    # ---- プロット: 3観測量 ----
    colors  = ["steelblue", "darkorange", "seagreen", "crimson"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    T_arr = T_list

    for L, c, mk in zip(L_list, colors, markers):
        lbl = f"$L={L}$"
        axes[0].plot(T_arr, avg_absm[L], marker=mk, ms=4, lw=1.2,
                     color=c, label=lbl)
        axes[1].plot(T_arr, chi[L],      marker=mk, ms=4, lw=1.2,
                     color=c, label=lbl)
        axes[2].plot(T_arr, U_L[L],      marker=mk, ms=4, lw=1.2,
                     color=c, label=lbl)

    for ax in axes:
        ax.axvline(TC_EXACT, color="k", lw=1.5, ls="--",
                   label=f"Onsager $T_c={TC_EXACT:.4f}$")
        ax.set_xlabel("$T / J$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"$\langle |m| \rangle$")
    axes[0].set_title("磁化")

    axes[1].set_ylabel(r"$\chi$")
    axes[1].set_title("磁化率")

    axes[2].set_ylabel("$U_L$")
    axes[2].set_title("Binderキュムラント")
    axes[2].axhline(2/3, color="gray", ls=":", lw=1, label="$2/3$")
    axes[2].axhline(0,   color="gray", ls=":", lw=1, label="$0$")

    plt.tight_layout()
    path = f"{outdir}/ising_observables.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"→ {path} 保存")

    # ---- T_c 推定（ビンダークロス） ----
    print()
    print("=" * 50)
    print("T_c 推定（ビンダークロス）")
    print("=" * 50)

    # 方法1: 全ペアの交差温度
    print("  [全ペアの交差温度]")
    from itertools import combinations
    Tc_pairs = []
    for L1, L2 in combinations(L_list, 2):
        U1 = np.array(U_L[L1])
        U2 = np.array(U_L[L2])
        diff = U1 - U2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            continue
        # T_c付近（1.9〜2.6）にある交差のみ採用
        for idx in sign_changes:
            frac = -diff[idx] / (diff[idx + 1] - diff[idx])
            T_cross = T_arr[idx] + frac * (T_arr[idx + 1] - T_arr[idx])
            if 1.9 < T_cross < 2.6:
                Tc_pairs.append(T_cross)
                print(f"  L={L1:3d} vs L={L2:3d}:  T_cross = {T_cross:.4f}")

    # 方法2: 全曲線の分散が最小になる温度（転移領域に限定）
    # 注意1: 低温側では全曲線が 2/3 に収束して分散がゼロになるため T > 2.1 に限定
    # 注意2: 最大の L は臨界スローダウンで曲線がノイジーになりやすいため除外
    print()
    L_for_var = L_list[:-1] if len(L_list) > 2 else L_list
    print(f"  [全曲線の分散が最小になる温度（L={L_for_var}, T > 2.1 に限定）]")
    U_matrix = np.array([U_L[L] for L in L_for_var])  # shape: (n_L, n_T)
    var_T = np.var(U_matrix, axis=0)                    # 各温度での L 間の分散
    transition_mask = T_arr > 2.1
    idx_min_local = np.argmin(var_T[transition_mask])
    idx_min = np.where(transition_mask)[0][idx_min_local]
    Tc_var = float(T_arr[idx_min])
    print(f"  分散最小点: T = {Tc_var:.4f}  (var = {var_T[idx_min]:.6f})")

    # まとめ
    print()
    all_estimates = Tc_pairs + [Tc_var]
    print()
    print(f"  分散最小   T_c = {Tc_var:.4f}")
    print(f"  厳密解         = {TC_EXACT:.4f}")
    err2 = abs(Tc_var - TC_EXACT)
    print(f"  誤差           = {err2:.4f}  ({100*err2/TC_EXACT:.2f}%)")

    # 分散の温度依存プロット（Binderクロスの鮮明化）
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_arr, var_T, "o-", color="steelblue")
    ax.axvline(TC_EXACT, color="k", ls="--", lw=1.5,
               label=f"Onsager $T_c={TC_EXACT:.4f}$")
    ax.axvline(Tc_var, color="crimson", ls=":", lw=1.5,
               label=f"var min $T={Tc_var:.4f}$")
    ax.set_xlabel("$T / J$")
    ax.set_ylabel("Var$[U_L]$ over $L$")
    ax.set_title("Binderキュムラントの L 間分散（最小点 = $T_c$ 推定）")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path_var = f"{outdir}/binder_variance.png"
    plt.savefig(path_var, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n→ {path_var} 保存")
    print()

    # ---- O(L^2) スケーリングテスト ----
    scaling_test(outdir=outdir)


if __name__ == "__main__":
    main()
