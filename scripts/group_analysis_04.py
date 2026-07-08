"""
Script 4: Group Analysis
Compares QPP results across all subjects
Creates group-level summary plots
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def load_all_results(task, subject_ids):
    """
    Load QPP results for all subjects for one task.
    Returns a pandas DataFrame with one row per subject.
    """
    rows = []
    for sub_id in subject_ids:
        info_path = f"results/{task}/{sub_id}_qpp_info.npy"
        if os.path.exists(info_path):
            info = np.load(info_path, allow_pickle=True).item()
            rows.append(info)
        else:
            print(f"  Missing: {sub_id} {task}")

    if len(rows) == 0:
        return None

    df = pd.DataFrame(rows)
    return df


def plot_group_summary(subject_ids):
    """
    Create group summary plots comparing rest vs emotion QPP.
    """
    print("\nCreating group summary plots...")

    df_rest = load_all_results("rest", subject_ids)
    df_fe   = load_all_results("fe",   subject_ids)

    if df_rest is None and df_fe is None:
        print("No results found yet!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Group QPP Analysis — NeuroEmo ds005700\n"
        f"({len(subject_ids)} subjects)",
        fontsize=13
    )

    # Plot 1: QPP occurrences per subject
    ax = axes[0]
    x = np.arange(len(subject_ids))
    if df_rest is not None:
        ax.bar(x - 0.2, df_rest['n_peaks'], width=0.4,
               label='Rest', color='steelblue', alpha=0.8)
    if df_fe is not None:
        ax.bar(x + 0.2, df_fe['n_peaks'], width=0.4,
               label='Emotion', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("sub-", "") for s in subject_ids],
        rotation=45, fontsize=8
    )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of QPP occurrences")
    ax.set_title("QPP Occurrences per Subject")
    ax.legend()

    # Plot 2: Mean correlation per subject
    ax = axes[1]
    if df_rest is not None:
        ax.plot(x, df_rest['mean_corr'], 'o-',
                color='steelblue', label='Rest', linewidth=1.5)
    if df_fe is not None:
        ax.plot(x, df_fe['mean_corr'], 's-',
                color='coral', label='Emotion', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("sub-", "") for s in subject_ids],
        rotation=45, fontsize=8
    )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Mean correlation (r)")
    ax.set_title("QPP Template Correlation")
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 3: Average interval per subject
    ax = axes[2]
    if df_rest is not None:
        ax.plot(x, df_rest['avg_interval'], 'o-',
                color='steelblue', label='Rest', linewidth=1.5)
    if df_fe is not None:
        ax.plot(x, df_fe['avg_interval'], 's-',
                color='coral', label='Emotion', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("sub-", "") for s in subject_ids],
        rotation=45, fontsize=8
    )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Average interval (s)")
    ax.set_title("QPP Average Interval")
    ax.legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/group_qpp_summary.png", dpi=150,
                bbox_inches='tight')
    plt.show()
    print("Saved → results/group_qpp_summary.png")

    # Print summary table
    print("\n" + "="*60)
    print("GROUP SUMMARY TABLE")
    print("="*60)
    if df_rest is not None:
        print("\nREST:")
        print(df_rest[['sub_id','n_peaks','mean_corr',
                        'avg_interval']].to_string(index=False))
    if df_fe is not None:
        print("\nEMOTION TASK:")
        print(df_fe[['sub_id','n_peaks','mean_corr',
                     'avg_interval']].to_string(index=False))

    # Save as CSV
    if df_rest is not None:
        df_rest.to_csv("results/group_rest_results.csv", index=False)
    if df_fe is not None:
        df_fe.to_csv("results/group_emotion_results.csv", index=False)
    print("\nSaved CSVs → results/group_rest_results.csv")
    print("           → results/group_emotion_results.csv")
