import argparse
import json
from itertools import combinations

import numpy as np


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def load_codes(strings):
    return np.asarray(
        [[int(bit) for bit in code] for code in strings], dtype=np.float64
    )


def z_from_llr_sum(probs_by_image, codes):
    p = np.clip(np.asarray(probs_by_image, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))
    llr_sum = llr.sum(axis=1)
    scores = ((2.0 * codes[None, :, :] - 1.0) * llr_sum[:, None, :]).sum(axis=2)
    norm = np.sqrt((llr * llr).sum(axis=(1, 2)))[:, None]
    return scores / norm


def load_attack(path):
    data = load_json(path)
    num_users = int(data['num_users'])
    images_per_user = int(data['images_per_user'])
    message_length = int(data['message_length'])
    probs = np.asarray(data['image_probs'], dtype=np.float64)
    user_ids = np.asarray(data['image_user_ids'], dtype=np.int64)
    image_indices = np.asarray(data['processed_image_indices'], dtype=np.int64)
    codes = load_codes(data['codes'])

    probs_by_user = np.zeros(
        (num_users, images_per_user, message_length), dtype=np.float64
    )
    for position, user_id in enumerate(user_ids):
        slot = int(image_indices[position] - user_id * images_per_user)
        probs_by_user[user_id, slot] = probs[position]

    return {
        'data': data,
        'codes': codes,
        'num_users': num_users,
        'images_per_user': images_per_user,
        'probs_by_user': probs_by_user,
    }


def threshold_rate(best_z, correct, thresholds):
    if len(best_z) == 0:
        return np.zeros(len(thresholds), dtype=np.float64)
    values = np.sort(best_z[correct])
    accepted = len(values) - np.searchsorted(values, thresholds, side='left')
    return accepted.astype(np.float64) / float(len(best_z))


def compute_curves(attack, thresholds):
    probs_by_user = attack['probs_by_user']
    codes = attack['codes']
    num_users = attack['num_users']
    images_per_user = attack['images_per_user']
    true_indices = np.arange(num_users)
    rows = {}

    for m in range(1, images_per_user + 1):
        best_z_rows = []
        true_z_rows = []
        margin_rows = []
        correct_rows = []
        for slots in combinations(range(images_per_user), m):
            z = z_from_llr_sum(probs_by_user[:, slots, :], codes)
            true_z = z[true_indices, true_indices]
            top1 = np.argmax(z, axis=1)
            best_z = z[true_indices, top1]
            other = z.copy()
            other[true_indices, true_indices] = -np.inf
            margin = true_z - np.max(other, axis=1)
            best_z_rows.append(best_z)
            true_z_rows.append(true_z)
            margin_rows.append(margin)
            correct_rows.append(top1 == true_indices)

        best_z = np.concatenate(best_z_rows)
        true_z = np.concatenate(true_z_rows)
        margin = np.concatenate(margin_rows)
        correct = np.concatenate(correct_rows)
        rates = threshold_rate(best_z, correct, thresholds)
        rows[m] = {
            'curve': rates,
            'mean_true_z': float(true_z.mean()),
            'mean_margin': float(margin.mean()),
            'min_true_z': float(true_z.min()),
            'min_best_z': float(best_z.min()),
            'top1': float(correct.mean()),
            'num_trials': int(len(best_z)),
            'fixed': {
                float(threshold): float(rate)
                for threshold, rate in zip(thresholds, rates)
            },
        }
    return rows


def threshold_value(row, threshold):
    return row['fixed'][float(threshold)]


def build_markdown(vqgan, simswap, thresholds):
    vq_image = vqgan['data']['per_image']
    ss_image = simswap['data']['per_image']
    lines = [
        '# VQGAN vs SimSwap-224',
        '',
        '## Per-image',
        '',
        '| Attack | BER | Exact 30-bit | Mean Hamming |',
        '|---|---:|---:|---:|',
        '| VQGAN | %.6f | %.4f | %.4f |' % (
            vq_image['ber'], vq_image['exact_match'], vq_image['mean_hamming']
        ),
        '| SimSwap-224 | %.6f | %.4f | %.4f |' % (
            ss_image['ber'], ss_image['exact_match'], ss_image['mean_hamming']
        ),
        '',
        '## Figure',
        '',
        'Panel (a): mean true-user evidence z; the two curves overlap because the '
        'difference is below 0.021 z. Panel (b) makes that tiny difference visible '
        'without offsetting the curves.',
        'Panels (c) and (d): empirical detection-rate step curves over a continuous '
        'Z=3..13 sweep for VQGAN and SimSwap-224. They are recomputed over 100 users '
        'and every non-empty image subset, not the three fixed points connected by lines.',
        '',
        '## Multi-image LLR, all subsets',
        '',
        '| m | VQGAN z | SimSwap-224 z | VQGAN Z=6 | SimSwap-224 Z=6 | '
        'VQGAN Z=8 | SimSwap-224 Z=8 | VQGAN Z=10 | SimSwap-224 Z=10 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for m in sorted(vqgan['curves'], key=int):
        v = vqgan['curves'][m]
        s = simswap['curves'][m]
        lines.append(
            '| %d | %.3f | %.3f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |' % (
                m,
                v['mean_true_z'],
                s['mean_true_z'],
                threshold_value(v, 6.0),
                threshold_value(s, 6.0),
                threshold_value(v, 8.0),
                threshold_value(s, 8.0),
                threshold_value(v, 10.0),
                threshold_value(s, 10.0),
            )
        )
    lines.extend([
        '',
        'The curves are empirical survival rates of the best LLR score over all '
        'non-empty image subsets. They are step functions, not fitted or '
        'interpolated detection curves. The reference vertical lines mark Z=6, 8, 10.',
    ])
    return '\n'.join(lines) + '\n'


def plot_curves(vqgan, simswap, thresholds, output_png):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 11.5,
        'axes.labelsize': 10.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    ms = sorted(vqgan['curves'], key=int)
    x_m = np.asarray([int(m) for m in ms])
    vqgan_z = np.asarray([vqgan['curves'][m]['mean_true_z'] for m in ms])
    simswap_z = np.asarray([simswap['curves'][m]['mean_true_z'] for m in ms])
    max_delta = float(np.max(np.abs(vqgan_z - simswap_z)))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.6, 4.9),
        gridspec_kw={'width_ratios': [1.12, 1.34, 1.34]},
    )

    axes[0].plot(
        x_m,
        vqgan_z,
        color='#4C78A8',
        linestyle='--',
        linewidth=4.2,
        marker='o',
        markersize=9,
        markerfacecolor='white',
        markeredgewidth=2.0,
        label='VQGAN',
        zorder=3,
    )
    axes[0].plot(
        x_m,
        simswap_z,
        color='#F58518',
        linestyle='-',
        linewidth=2.2,
        marker='s',
        markersize=6,
        label='SimSwap-224',
        zorder=4,
    )
    axes[0].set_title('Evidence accumulation')
    axes[0].set_xlabel('Number of leaked images m')
    axes[0].set_ylabel('Mean true-user evidence z')
    axes[0].set_xticks(x_m)
    axes[0].set_ylim(
        min(vqgan_z.min(), simswap_z.min()) - 0.5,
        max(vqgan_z.max(), simswap_z.max()) + 0.5,
    )
    axes[0].grid(alpha=0.28, axis='y')
    axes[0].legend(fontsize=8.5, loc='lower right', frameon=False)
    axes[0].text(
        0.04,
        0.96,
        f'Curves overlap\nmax |delta z| = {max_delta:.3f}',
        transform=axes[0].transAxes,
        fontsize=8.5,
        color='#333333',
        verticalalignment='top',
        bbox={'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': 0.92},
    )

    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2']
    for axis, title, rows in (
        (axes[1], 'VQGAN: continuous threshold sweep', vqgan['curves']),
        (axes[2], 'SimSwap-224: continuous threshold sweep', simswap['curves']),
    ):
        for index, m in enumerate(ms):
            axis.plot(
                thresholds,
                rows[m]['curve'],
                drawstyle='steps-post',
                color=colors[index % len(colors)],
                linewidth=2.25,
                label=f'm={m}',
            )
        for reference in (6.0, 8.0, 10.0):
            axis.axvline(reference, color='#8A8A8A', linestyle=':', linewidth=1.0)
        axis.text(6.08, 1.015, 'Z=6', fontsize=7.5, color='#666666', va='bottom')
        axis.text(8.08, 1.015, 'Z=8', fontsize=7.5, color='#666666', va='bottom')
        axis.text(10.08, 1.015, 'Z=10', fontsize=7.5, color='#666666', va='bottom')
        axis.set_title(title)
        axis.set_xlabel('Detection threshold Z')
        axis.set_ylabel('Detection rate')
        axis.set_xlim(float(thresholds[0]), float(thresholds[-1]))
        axis.set_ylim(-0.03, 1.07)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0%}'))
        axis.grid(alpha=0.25, axis='y')
        axis.legend(
            title='leaked images',
            fontsize=8,
            title_fontsize=8,
            loc='lower left',
            frameon=False,
            ncol=2,
        )

    fig.suptitle(
        'Multi-image watermark evidence and threshold sensitivity',
        fontsize=14,
        fontweight='bold',
        y=0.99,
    )
    fig.text(
        0.5,
        0.018,
        'Empirical step curves over 100 users and all non-empty subsets; '
        'dotted lines mark Z=6, 8, 10.',
        ha='center',
        fontsize=8.5,
        color='#444444',
    )
    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.955])
    fig.savefig(output_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_curves_4panel(vqgan, simswap, thresholds, output_png):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 11.0,
        'axes.labelsize': 10.0,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    ms = sorted(vqgan['curves'], key=int)
    x_m = np.asarray([int(m) for m in ms])
    vqgan_z = np.asarray([vqgan['curves'][m]['mean_true_z'] for m in ms])
    simswap_z = np.asarray([simswap['curves'][m]['mean_true_z'] for m in ms])
    delta_z = vqgan_z - simswap_z
    max_delta = float(np.max(np.abs(delta_z)))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    ax_a.plot(
        x_m,
        vqgan_z,
        color='#4C78A8',
        linestyle='--',
        linewidth=4.2,
        marker='o',
        markersize=9,
        markerfacecolor='white',
        markeredgewidth=2.0,
        label='VQGAN',
        zorder=3,
    )
    ax_a.plot(
        x_m,
        simswap_z,
        color='#F58518',
        linestyle='-',
        linewidth=2.0,
        marker='s',
        markersize=6,
        label='SimSwap-224',
        zorder=4,
    )
    ax_a.set_title('(a) Evidence accumulation')
    ax_a.set_xlabel('Number of leaked images m')
    ax_a.set_ylabel('Mean true-user evidence z')
    ax_a.set_xticks(x_m)
    ax_a.set_ylim(
        min(vqgan_z.min(), simswap_z.min()) - 0.5,
        max(vqgan_z.max(), simswap_z.max()) + 0.5,
    )
    ax_a.grid(alpha=0.28, axis='y')
    ax_a.legend(fontsize=8.5, loc='lower right', frameon=False)
    ax_a.text(
        0.04,
        0.96,
        f'Curves overlap\nmax |delta z| = {max_delta:.3f}',
        transform=ax_a.transAxes,
        fontsize=8.5,
        color='#333333',
        verticalalignment='top',
        bbox={'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': 0.92},
    )

    ax_b.bar(x_m, delta_z, width=0.55, color='#4C78A8', edgecolor='#2F5D82')
    ax_b.axhline(0.0, color='#333333', linewidth=0.8)
    ax_b.set_title('(b) Difference is below 0.021 z')
    ax_b.set_xlabel('Number of leaked images m')
    ax_b.set_ylabel('Delta z (VQGAN - SimSwap)')
    ax_b.set_xticks(x_m)
    ax_b.set_ylim(-0.004, max(float(delta_z.max()) * 1.35, 0.024))
    ax_b.grid(alpha=0.28, axis='y')
    for x_value, y_value in zip(x_m, delta_z):
        ax_b.text(
            x_value,
            y_value + 0.0006,
            f'{y_value:+.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#333333',
        )
    ax_b.text(
        0.04,
        0.96,
        'Positive only means VQGAN z is slightly higher;\n'
        'detection at Z=6/8/10 is identical.',
        transform=ax_b.transAxes,
        fontsize=8.0,
        color='#333333',
        verticalalignment='top',
        bbox={'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': 0.92},
    )

    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2']
    for axis, title, rows in (
        (ax_c, '(c) VQGAN: empirical threshold curves', vqgan['curves']),
        (ax_d, '(d) SimSwap-224: empirical threshold curves', simswap['curves']),
    ):
        for index, m in enumerate(ms):
            axis.plot(
                thresholds,
                rows[m]['curve'],
                drawstyle='steps-post',
                color=colors[index % len(colors)],
                linewidth=2.15,
                label=f'm={m}',
            )
        for reference in (6.0, 8.0, 10.0):
            axis.axvline(reference, color='#8A8A8A', linestyle=':', linewidth=1.0)
        axis.text(6.08, 1.015, 'Z=6', fontsize=7.5, color='#666666', va='bottom')
        axis.text(8.08, 1.015, 'Z=8', fontsize=7.5, color='#666666', va='bottom')
        axis.text(10.08, 1.015, 'Z=10', fontsize=7.5, color='#666666', va='bottom')
        axis.set_title(title)
        axis.set_xlabel('Detection threshold Z')
        axis.set_ylabel('Detection rate')
        axis.set_xlim(float(thresholds[0]), float(thresholds[-1]))
        axis.set_ylim(-0.03, 1.07)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0%}'))
        axis.grid(alpha=0.25, axis='y')
        axis.legend(
            title='leaked images',
            fontsize=8,
            title_fontsize=8,
            loc='lower left',
            frameon=False,
            ncol=2,
        )

    fig.suptitle(
        'Multi-image watermark evidence and threshold sensitivity',
        fontsize=14,
        fontweight='bold',
        y=0.985,
    )
    fig.text(
        0.5,
        0.016,
        'Panels (c) and (d) are empirical step curves over 100 users and all '
        'non-empty subsets. Dotted lines mark Z=6, 8, 10.',
        ha='center',
        fontsize=8.5,
        color='#444444',
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    fig.savefig(output_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def build_markdown_three(vqgan, simswap, inswapper):
    attacks = (
        ('VQGAN', vqgan),
        ('SimSwap-224', simswap),
        ('InSwapper-128', inswapper),
    )
    lines = [
        '# VQGAN vs SimSwap-224 vs InSwapper-128',
        '',
        '## Per-image decoding',
        '',
        '| Attack | BER | Exact 30-bit | Mean Hamming |',
        '|---|---:|---:|---:|',
    ]
    for label, attack in attacks:
        image = attack['data']['per_image']
        lines.append(
            '| %s | %.6f | %.4f | %.4f |' % (
                label,
                image['ber'],
                image['exact_match'],
                image['mean_hamming'],
            )
        )
    lines.extend([
        '',
        '## Multi-image LLR',
        '',
        '| m | VQGAN z | SimSwap-224 z | InSwapper-128 z | VQGAN margin | SimSwap-224 margin | InSwapper-128 margin |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for m in sorted(vqgan['curves'], key=int):
        rows = [attack['curves'][m] for _, attack in attacks]
        lines.append(
            '| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
                m,
                rows[0]['mean_true_z'],
                rows[1]['mean_true_z'],
                rows[2]['mean_true_z'],
                rows[0]['mean_margin'],
                rows[1]['mean_margin'],
                rows[2]['mean_margin'],
            )
        )
    lines.extend([
        '',
        'Panel (a) shows the absolute evidence curves; panel (b) zooms the tiny '
        'differences from VQGAN instead of claiming that overlapping curves show a '
        'material attack ranking. Panels (c)-(e) are empirical threshold sweeps for '
        'each attack. Panel (f) reports the single-image bit-error rate.',
        '',
        'The three attacks produce nearly identical evidence. The defensible claim is '
        'that the watermark remains traceable under all three, not that one attack is '
        'significantly stronger.',
    ])
    return '\n'.join(lines) + '\n'


def plot_curves_6panel(vqgan, simswap, inswapper, thresholds, output_png):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 10.5,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
    })

    attacks = (
        ('VQGAN', vqgan, '#4C78A8', 'o', '--'),
        ('SimSwap-224', simswap, '#F58518', 's', '-'),
        ('InSwapper-128', inswapper, '#2F855A', '^', '-'),
    )
    ms = sorted(vqgan['curves'], key=int)
    x_m = np.asarray([int(m) for m in ms])
    z_values = {
        label: np.asarray([attack['curves'][m]['mean_true_z'] for m in ms])
        for label, attack, *_ in attacks
    }
    reference = z_values['VQGAN']

    fig, axes = plt.subplots(2, 3, figsize=(17.2, 8.6))
    ax_a, ax_b, ax_f = axes[0]
    ax_c, ax_d, ax_e = axes[1]

    for label, _, color, marker, linestyle in attacks:
        ax_a.plot(
            x_m,
            z_values[label],
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            label=label,
        )
    max_delta = max(
        float(np.max(np.abs(z_values[label] - reference)))
        for label, *_ in attacks
        if label != 'VQGAN'
    )
    ax_a.set_title('(a) Evidence accumulation')
    ax_a.set_ylabel('Mean true-user z')
    ax_a.text(
        0.04,
        0.96,
        f'All curves overlap\nmax |delta z| = {max_delta:.3f}',
        transform=ax_a.transAxes,
        va='top',
        fontsize=8.2,
        bbox={'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': 0.92},
    )

    width = 0.34
    offsets = (-width / 2.0, width / 2.0)
    for index, (label, _, color, _, _) in enumerate(
        attack for attack in attacks if attack[0] != 'VQGAN'
    ):
        ax_b.bar(
            x_m + offsets[index],
            z_values[label] - reference,
            width=width,
            color=color,
            alpha=0.88,
            label=f'{label} - VQGAN',
        )
    ax_b.axhline(0.0, color='#333333', linewidth=0.8)
    ax_b.set_title('(b) Zoomed difference from VQGAN')
    ax_b.set_ylabel('Delta mean z')
    ax_b.legend(frameon=False, fontsize=8)

    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2']
    for axis, title, attack in (
        (ax_c, '(c) VQGAN threshold sweep', vqgan),
        (ax_d, '(d) SimSwap-224 threshold sweep', simswap),
        (ax_e, '(e) InSwapper-128 threshold sweep', inswapper),
    ):
        for index, m in enumerate(ms):
            axis.plot(
                thresholds,
                attack['curves'][m]['curve'],
                drawstyle='steps-post',
                color=colors[index],
                linewidth=2.0,
                label=f'm={m}',
            )
        for reference_threshold in (6.0, 8.0, 10.0):
            axis.axvline(
                reference_threshold,
                color='#8A8A8A',
                linestyle=':',
                linewidth=0.9,
            )
        axis.set_title(title)
        axis.set_ylabel('Detection rate')
        axis.set_xlim(float(thresholds[0]), float(thresholds[-1]))
        axis.set_ylim(-0.03, 1.07)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0%}'))
        axis.legend(
            title='leaked images',
            fontsize=7.5,
            title_fontsize=7.5,
            loc='lower left',
            frameon=False,
            ncol=2,
        )

    labels = [label for label, *_ in attacks]
    ber = [100.0 * attack['data']['per_image']['ber'] for _, attack, *_ in attacks]
    bar_colors = [color for _, _, color, _, _ in attacks]
    bars = ax_f.bar(labels, ber, color=bar_colors, width=0.55)
    ax_f.set_title('(f) Single-image bit-error rate')
    ax_f.set_ylabel('BER (percent)')
    ax_f.set_ylim(0.0, max(max(ber) * 1.45, 0.02))
    for bar, value in zip(bars, ber):
        ax_f.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f'{value:.4f}%',
            ha='center',
            va='bottom',
            fontsize=8.5,
        )

    for axis in (ax_a, ax_b):
        axis.set_xlabel('Number of leaked images m')
        axis.set_xticks(x_m)
        axis.grid(alpha=0.25, axis='y')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    for axis in (ax_c, ax_d, ax_e):
        axis.set_xlabel('Detection threshold Z')
        axis.grid(alpha=0.23, axis='y')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    ax_a.legend(frameon=False, fontsize=8, loc='lower right')

    fig.suptitle(
        'Multi-image watermark evidence under three attacks',
        fontsize=14,
        fontweight='bold',
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        '100 users x 5 images, every non-empty subset. Dotted lines mark Z=6, 8, 10.',
        ha='center',
        fontsize=8.5,
        color='#475467',
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    fig.savefig(output_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_json', required=True)
    parser.add_argument('--simswap_json', required=True)
    parser.add_argument('--inswapper_json', default='')
    parser.add_argument('--vqgan_llr', default='')
    parser.add_argument('--simswap_llr', default='')
    parser.add_argument('--output_md', required=True)
    parser.add_argument('--output_png', required=True)
    args = parser.parse_args()

    thresholds = np.linspace(3.0, 13.0, 201)
    vqgan = load_attack(args.vqgan_json)
    simswap = load_attack(args.simswap_json)
    inswapper = load_attack(args.inswapper_json) if args.inswapper_json else None
    vqgan['curves'] = compute_curves(vqgan, thresholds)
    simswap['curves'] = compute_curves(simswap, thresholds)

    if inswapper is not None:
        inswapper['curves'] = compute_curves(inswapper, thresholds)
        text = build_markdown_three(vqgan, simswap, inswapper)
    else:
        text = build_markdown(vqgan, simswap, thresholds)
    print(text)
    with open(args.output_md, 'w', encoding='utf-8') as handle:
        handle.write(text)
    if inswapper is not None:
        plot_curves_6panel(vqgan, simswap, inswapper, thresholds, args.output_png)
    else:
        plot_curves_4panel(vqgan, simswap, thresholds, args.output_png)
    print(f'Wrote {args.output_md}')
    print(f'Wrote {args.output_png}')


if __name__ == '__main__':
    main()
