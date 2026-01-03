import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 确保使用 TkAgg 后端
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import vonmises


def parse_float_list(text: str, fallback: list[float]) -> list[float]:
    if not text.strip():
        return list(fallback)
    parts = text.replace(",", " ").split()
    return [float(p) for p in parts]


def normalize_weights(weights: list[float]) -> list[float]:
    arr = np.array(weights, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("权重列表不能为空。")
    if np.any(arr < 0):
        raise ValueError("权重必须非负。")
    total = arr.sum()
    if total <= 0:
        raise ValueError("权重和必须大于 0。")
    return (arr / total).tolist()


def mixture_pdf(theta: np.ndarray, mus: list[float], kappas: list[float], weights: list[float]) -> np.ndarray:
    pdf = np.zeros_like(theta, dtype=float)
    for mu, kappa, w in zip(mus, kappas, weights):
        pdf += w * vonmises.pdf(theta, kappa=kappa, loc=mu)
    return pdf


def figure_exists(fig) -> bool:
    """检查figure是否仍然存在"""
    try:
        return plt.fignum_exists(fig.number)
    except:
        return False


def main() -> None:
    print("=" * 60)
    print("  Von Mises 混合分布实时绘制工具")
    print("=" * 60)
    print("输入 q 退出，直接回车沿用当前/默认值。")
    print()

    theta = np.linspace(-np.pi, np.pi, 800)

    # 默认值
    K = 4  # 峰数
    kappa_default = 10.0
    mu_deg_list = [0.0, 90.0, 180.0, 270.0]
    kappas = [kappa_default] * K
    weights = [1.0] * K

    plt.ion()
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(1, 2, 1)
    ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

    # 设置窗口标题（在创建时设置一次）
    try:
        fig.canvas.manager.set_window_title("Von Mises Mixture")
    except:
        pass

    plt.show(block=False)

    while True:
        # 检查窗口是否已关闭
        if not figure_exists(fig):
            print("\n窗口已关闭，退出程序。")
            break

        print("-" * 40)

        # Step 1: 输入峰数 K
        k_text = input(f"峰数 K (当前: {K}, 示例: 1/2/4): ").strip()
        if k_text.lower().startswith("q"):
            break
        if k_text:
            try:
                K = int(k_text)
                if K < 1:
                    raise ValueError("K 必须 >= 1")
                # 自动生成均匀分布的 mu
                mu_deg_list = [360.0 * i / K for i in range(K)]
                kappas = [kappa_default] * K
                weights = [1.0] * K
                print(f"  -> 自动生成 mu: {mu_deg_list}")
            except ValueError as e:
                print(f"输入错误: {e}")
                continue

        # Step 2: 输入 kappa（可以输入单个值应用到所有峰）
        kappa_text = input(f"kappa (当前: {kappas[0]}, 输入单值或列表): ").strip()
        if kappa_text.lower().startswith("q"):
            break
        if kappa_text:
            try:
                kappa_values = parse_float_list(kappa_text, kappas)
                if len(kappa_values) == 1:
                    # 单值应用到所有峰
                    kappas = [kappa_values[0]] * K
                elif len(kappa_values) == K:
                    kappas = kappa_values
                else:
                    raise ValueError(f"kappa 数量需为 1 或 {K}")
            except ValueError as e:
                print(f"输入错误: {e}")
                continue

        # Step 3: 可选自定义 mu
        mu_text = input(f"自定义 mu (角度，可空用默认均匀分布): ").strip()
        if mu_text.lower().startswith("q"):
            break
        if mu_text:
            try:
                mu_deg_list = parse_float_list(mu_text, mu_deg_list)
                if len(mu_deg_list) != K:
                    print(f"警告: mu 数量 ({len(mu_deg_list)}) != K ({K})，自动调整 K")
                    K = len(mu_deg_list)
                    if len(kappas) != K:
                        kappas = [kappas[0]] * K
                    weights = [1.0] * K
            except ValueError as e:
                print(f"输入错误: {e}")
                continue

        # Step 4: 可选权重
        weight_text = input(f"权重 (可空，默认均分): ").strip()
        if weight_text.lower().startswith("q"):
            break
        if weight_text:
            try:
                weights_input = parse_float_list(weight_text, weights)
                if len(weights_input) != K:
                    raise ValueError(f"权重数量需为 {K}")
                weights = normalize_weights(weights_input)
            except ValueError as e:
                print(f"输入错误: {e}")
                continue
        else:
            weights = normalize_weights([1.0] * K)

        # 检查窗口是否仍然存在
        if not figure_exists(fig):
            print("\n窗口已关闭，退出程序。")
            break

        # 绘图
        try:
            mus = np.deg2rad(mu_deg_list).tolist()
            pdf_mix = mixture_pdf(theta, mus, kappas, weights)

            ax.cla()
            ax_polar.cla()

            colors = cm.plasma(np.linspace(0.1, 0.9, len(mus)))
            for idx, (mu_deg, mu_rad, kappa, w, color) in enumerate(zip(mu_deg_list, mus, kappas, weights, colors)):
                comp_pdf = vonmises.pdf(theta, kappa=kappa, loc=mu_rad)
                ax.plot(theta, comp_pdf, color=color, linestyle="--", linewidth=1.2, alpha=0.7,
                        label=rf"峰{idx+1}: $\mu={mu_deg:.0f}^\circ$, $\kappa={kappa:.1f}$")
                ax_polar.plot(theta, comp_pdf, color=color, linestyle="--", linewidth=1.0, alpha=0.6)

            ax.plot(theta, pdf_mix, color="black", linewidth=2.0, label="混合分布")
            ax_polar.plot(theta, pdf_mix, color="black", linewidth=2.0)

            ax.set_xlabel(r"$\theta$ (radians)")
            ax.set_ylabel("Probability density")
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks(np.deg2rad([-180, -90, 0, 90, 180]))
            ax.set_xticklabels([r"$-180^\circ$", r"$-90^\circ$", r"$0^\circ$", r"$90^\circ$", r"$180^\circ$"])
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
            ax.set_ylim(0, None)
            ax.set_title(f"K={K}, kappa={kappas[0]:.1f}" if len(set(kappas)) == 1 else f"K={K}")

            ax_polar.set_title("Polar view", fontsize=11)
            ax_polar.grid(True, alpha=0.3, linestyle="--")
            ax_polar.set_theta_zero_location("N")  # 0度在顶部
            ax_polar.set_theta_direction(-1)  # 顺时针
            ax_polar.set_ylim(0, None)

            fig.tight_layout()
            plt.pause(0.1)

        except Exception as e:
            print(f"绘图错误: {e}")
            if not figure_exists(fig):
                break

    plt.ioff()
    try:
        plt.close("all")
    except:
        pass
    print("程序退出。")


if __name__ == "__main__":
    main()
