import argparse
import json

import numpy as np


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean_json', required=True)
    parser.add_argument('--median_json', required=True)
    parser.add_argument('--mosaic_json', required=True)
    parser.add_argument('--output_png', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--attack_name', default='SimSwap224')
    parser.add_argument('--title', default='SimSwap-224 after mean / median / mosaic collusion')
    parser.add_argument('--top1_style', choices=['line', 'bar'], default='line')
    parser.add_argument(
        '--ci_style',
        choices=['errorbar', 'band', 'none'],
        default='errorbar',
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = [
        ('mean', args.mean_json, '#4C78A8', 'o'),
        ('median', args.median_json, '#F58518', 's'),
        ('mosaic', args.mosaic_json, '#E45756', '^'),
    ]
    data = {}
    for mix, path, color, marker in sources:
        data[mix] = load_json(path)

    ks = [2, 3, 4, 5]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))

    bar_handles = []
    for mix_index, (mix, _, color, marker) in enumerate(sources):
        attack = data[mix]['conditions'][args.attack_name]
        top1 = []
        low_err = []
        high_err = []
        detect = []
        false_accuse = []
        for k in ks:
            row = attack[str(k)]
            rate = row['top1']
            ci_low, ci_high = row['top1_ci95']
            top1.append(rate)
            low_err.append(max(0.0, rate - ci_low))
            high_err.append(max(0.0, ci_high - rate))
            op = row['operating_points']['3.0']
            detect.append(op['detect'])
            false_accuse.append(op['false_accuse'])

        if args.top1_style == 'bar':
            offset = (mix_index - 1) * 0.24
            bars = axes[0].bar(
                np.asarray(ks) + offset,
                top1,
                width=0.24,
                color=color,
                alpha=0.9,
                label=mix,
            )
            bar_handles.append(bars[0])
            for bar, value in zip(bars, top1):
                if value < 0.995:
                    axes[0].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        value - 0.055,
                        f'{100.0 * value:.0f}%',
                        ha='center',
                        va='top',
                        fontsize=7.0,
                        color='white',
                        fontweight='bold',
                    )
        elif args.ci_style == 'none':
            axes[0].plot(
                ks,
                top1,
                marker + '-',
                color=color,
                linewidth=1.8,
                markersize=6,
                label=mix,
            )
        elif args.ci_style == 'band':
            axes[0].plot(
                ks,
                top1,
                marker + '-',
                color=color,
                linewidth=1.8,
                markersize=6,
                label=mix,
            )
            axes[0].fill_between(
                ks,
                np.asarray(top1) - np.asarray(low_err),
                np.asarray(top1) + np.asarray(high_err),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        else:
            axes[0].errorbar(
                ks,
                top1,
                yerr=np.asarray([low_err, high_err]),
                fmt=marker + '-',
                color=color,
                capsize=3,
                linewidth=1.8,
                markersize=6,
                label=mix,
            )
        axes[1].plot(
            ks,
            detect,
            marker + '-',
            color=color,
            linewidth=1.8,
            markersize=6,
            label=mix + ' detection',
        )
        axes[1].plot(
            ks,
            false_accuse,
            marker + '--',
            color=color,
            linewidth=1.2,
            markersize=4,
            alpha=0.55,
        )

    axes[0].set_title('(a) Soft top-1 attribution')
    axes[0].set_ylabel('Top-1 accuracy')
    axes[1].set_title('(b) Z=3.0 operating point')
    axes[1].set_ylabel('Rate')
    if args.top1_style == 'bar':
        fig.legend(
            bar_handles,
            [mix for mix, _, _, _ in sources],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.93),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
    else:
        axes[0].legend(fontsize=8, frameon=False)
        axes[1].legend(fontsize=8, frameon=False)

    for axis in axes:
        axis.set_xlabel('Number of colluders k')
        axis.set_xticks(ks)
        axis.set_ylim(-0.02, 1.05)
        axis.grid(alpha=0.25)

    num_groups = data[sources[0][0]]['num_groups']
    fig.suptitle(args.title, fontsize=12)
    if args.top1_style == 'bar' or args.ci_style == 'none':
        interval_note = (
            f'{num_groups} groups per k. Unlabeled bars are 100%. '
            '95% Wilson CI at 100% is [72.2%, 100%]. '
        )
    else:
        interval_note = f'{num_groups} groups per point. Error bars: 95% Wilson CI. '
    fig.text(
        0.5,
        0.015,
        interval_note
        + 'In (b), solid lines are detection and dashed lines are false accusation.',
        ha='center',
        fontsize=8,
        color='#444444',
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
    fig.savefig(args.output_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

    compact = {
        'attack': args.attack_name,
        'threshold': 3.0,
        'num_groups': 100,
        'ks': ks,
        'mixes': {
            mix: data[mix]['conditions'][args.attack_name] for mix, _, _, _ in sources
        },
    }
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(compact, handle, indent=2)
    print('saved', args.output_png)
    print('saved', args.output_json)


if __name__ == '__main__':
    main()
