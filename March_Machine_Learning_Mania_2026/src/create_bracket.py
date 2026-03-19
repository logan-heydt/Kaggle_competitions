import pandas as pd

# Load data
seeds_df = pd.read_csv('../data/MNCAATourneySeeds.csv')
teams_df = pd.read_csv('../data/MTeams.csv')
slots_df = pd.read_csv('../data/MNCAATourneySlots.csv')
sub_df = pd.read_csv('../submissions/submission_advanced_2026.csv')

# Filter to 2026
seeds_2026 = seeds_df[seeds_df['Season'] == 2026].set_index('Seed')['TeamID'].to_dict()
slots_2026 = slots_df[slots_df['Season'] == 2026][['Slot', 'StrongSeed', 'WeakSeed']].values.tolist()
team_names = teams_df.set_index('TeamID')['TeamName'].to_dict()

# Build probability lookup
prob_lookup = {}
for _, row in sub_df.iterrows():
    parts = row['ID'].split('_')
    if parts[0] == '2026':
        t1, t2 = int(parts[1]), int(parts[2])
        prob_lookup[(t1, t2)] = row['Pred']  # prob t1 beats t2

def get_win_prob(t1, t2):
    """Probability that t1 beats t2."""
    if (t1, t2) in prob_lookup:
        return prob_lookup[(t1, t2)]
    elif (t2, t1) in prob_lookup:
        return 1 - prob_lookup[(t2, t1)]
    else:
        return 0.5  # fallback

def pick_winner(t1, t2):
    """Return winner (team with highest win prob)."""
    p = get_win_prob(t1, t2)
    return t1 if p >= 0.5 else t2

# slot_winners maps slot -> winning TeamID
slot_winners = {}

# Seed -> TeamID resolution (handles play-in slots)
def resolve_seed(seed):
    """Return TeamID for a seed, or None if it's a slot reference."""
    if seed in seeds_2026:
        return seeds_2026[seed]
    return None

# Process slots in order (play-ins first, then R1, R2, etc.)
# Sort: play-ins (no R prefix) first, then by round number
def slot_sort_key(row):
    slot = row[0]
    if slot.startswith('R'):
        return (int(slot[1]), slot)
    return (0, slot)  # play-in games

slots_sorted = sorted(slots_2026, key=slot_sort_key)

print("=" * 70)
print("2026 NCAA TOURNAMENT BRACKET - MODEL PREDICTIONS")
print("=" * 70)

round_names = {
    0: "First Four (Play-In)",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "National Championship"
}

current_round = -1

for slot, strong_seed, weak_seed in slots_sorted:
    # Determine round
    if slot.startswith('R'):
        rnd = int(slot[1])
    else:
        rnd = 0

    if rnd != current_round:
        current_round = rnd
        print(f"\n{'─' * 70}")
        print(f"  {round_names.get(rnd, f'Round {rnd}')}")
        print(f"{'─' * 70}")

    # Resolve teams
    if strong_seed in slot_winners:
        t1 = slot_winners[strong_seed]
    else:
        t1 = resolve_seed(strong_seed)

    if weak_seed in slot_winners:
        t2 = slot_winners[weak_seed]
    else:
        t2 = resolve_seed(weak_seed)

    if t1 is None or t2 is None:
        print(f"  [{slot}] Could not resolve: {strong_seed} vs {weak_seed}")
        continue

    winner = pick_winner(t1, t2)
    loser = t2 if winner == t1 else t1
    prob = get_win_prob(winner, loser)

    slot_winners[slot] = winner

    t1_name = team_names.get(t1, str(t1))
    t2_name = team_names.get(t2, str(t2))
    winner_name = team_names.get(winner, str(winner))

    print(f"  {slot:<8}  {t1_name:<20} vs {t2_name:<20}  →  {winner_name} ({prob:.1%})")

print(f"\n{'=' * 70}")
print(f"  CHAMPION: {team_names.get(slot_winners.get('R6CH'), 'Unknown')}")
print(f"{'=' * 70}\n")

# Also save bracket results to CSV
results = []
for slot, strong_seed, weak_seed in slots_sorted:
    if slot.startswith('R'):
        rnd = int(slot[1])
    else:
        rnd = 0

    if strong_seed in slot_winners:
        t1 = slot_winners[strong_seed]
    else:
        t1 = resolve_seed(strong_seed)

    if weak_seed in slot_winners:
        t2 = slot_winners[weak_seed]
    else:
        t2 = resolve_seed(weak_seed)

    if t1 is None or t2 is None:
        continue

    winner = pick_winner(t1, t2)
    loser = t2 if winner == t1 else t1
    prob = get_win_prob(winner, loser)

    results.append({
        'Slot': slot,
        'Round': rnd,
        'Team1': team_names.get(t1, str(t1)),
        'Team2': team_names.get(t2, str(t2)),
        'Winner': team_names.get(winner, str(winner)),
        'WinProb': round(prob, 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('../submissions/bracket_2026.csv', index=False)
print("Bracket saved to submissions/bracket_2026.csv")
