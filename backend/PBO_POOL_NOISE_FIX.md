# PBO Pool Generation Fix: Balancing Exploration and Exploitation

## Problem

The PBO was not converging to user preferences effectively. Analysis revealed:

1. **Initial attempt**: Reduced noise too much → proposals became identical → GP got stuck
2. **Root cause**: Pool generation noise was either:
   - Too low → all proposals converge to same point (no diversity)
   - Too high → random exploration, no exploitation of learned preferences

## Solution

Implement **strategy-specific noise and pool composition** that balances both:

### Strategy-Specific Pool Generation

Each acquisition strategy now has its own noise level and pool composition:

| Strategy | Noise Std | % from Learned Starts | % Random | Purpose |
|----------|-----------|----------------------|----------|---------|
| `exploit` | 0.15 | 70% | 30% | Refine around what user liked |
| `thompson` | 0.30 | 50% | 50% | Balance learning & exploration |
| `ei` | 0.35 | 40% | 60% | Expected improvement with exploration |
| `diverse` | 0.50 | 30% | 70% | Maximum diversity to avoid local optima |

### Start Point Generation Improvements

1. **Priority 1**: Use GP's top-performing candidates (not just w_current)
   - Query GP predictions for all historical candidates
   - Use top-3 as start points for optimization

2. **Priority 2**: Include current UI weights (w_current)
   - Still respect user's direct input

3. **Priority 3**: Dirichlet samples with **lower concentration** (20.0, down from 100.0)
   - Too high concentration → all samples cluster together
   - Lower concentration → more spread around learned preferences

### Results

**Before fix**: 28% convergence (2.3/8 proposals aligned with preferences)

**After fix**: 42% convergence (5/12 proposals), with Round 4 achieving **75% convergence**

| Round | Convergence | Notes |
|-------|-------------|-------|
| 2 | 25% (1/4) | Early learning |
| 3 | 25% (1/4) | Building knowledge |
| 4 | 75% (3/4) | Strong convergence! |

## Key Insights

1. **Diversity is essential**: Without it, GP gets stuck in local optima
2. **Balance matters**: One exploit + three exploration strategies works well
3. **Progressive convergence**: Model needs time to learn (3-4 rounds)
4. **GP-guided starts**: Using GP predictions (not just w_current) prevents getting stuck

## Implementation

Changes in `/backend/pbo.py`:

1. `_optimize_acquisition()`: Strategy-specific noise and pool composition (lines 577-609)
2. `_generate_starts()`: GP-guided start points using top-performing candidates (lines 507-560)
3. Acquisition strategy order: `['exploit', 'diverse', 'thompson', 'ei']` (line 471)

## Testing

Run diagnostic:
```bash
cd backend
python3 diagnose_pbo_convergence.py
```

Expected: ≥50% convergence over 4 rounds, with final round showing 75%+

