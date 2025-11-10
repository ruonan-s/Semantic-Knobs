# PBO Debug Quick Reference Card

## 🚀 Quick Commands

### Check PBO State
```bash
curl "http://localhost:8765/api/pbo/debug-state?session_id=<SESSION>&stage=impression" | python -m json.tool
```

### Watch Server Logs
```bash
cd backend && conda activate apl && python server.py 2>&1 | grep -E "\[PBO|REFINER"
```

### Compare Round Weights
```python
import json, numpy as np

with open("sessions/<SESSION>/impression_refinement/round_1/weights.json") as f:
    r1 = json.load(f)
with open("sessions/<SESSION>/impression_refinement/round_2/weights.json") as f:
    r2 = json.load(f)

for i in range(4):
    identical = np.allclose(r1['proposals'][i], r2['proposals'][i], atol=1e-6)
    print(f"Proposal {i}: {'❌ IDENTICAL (BUG)' if identical else '✅ Different (correct)'}")
```

---

## 🔍 What to Look For

### ✅ Correct Flow (Full Pipeline)
```
Round 1:
[PBO INIT] or [generate-stage-refinement]
  → candidates: 0, duels: 0, fitted: False
  → 4 proposals (cold start)

Round 2:
[PBO REFINE NEXT ROUND]
  → Before: candidates: 0, duels: 0
  → After recording: candidates: 4, duels: 3
  → After propose: fitted: True
  → 4 DIFFERENT proposals
```

### ❌ Wrong Flow (Bug)
```
Round 1:
[PBO PROPOSE]
  → candidates: 0, duels: 0

Round 2:
[PBO PROPOSE]  ← WRONG! Should be REFINE-NEXT-ROUND
  → candidates: 0, duels: 0  ← Still 0! No learning!
```

---

## 📊 Expected PBO State Evolution

| Round | Candidates | Duels | Fitted | Proposal Type |
|-------|-----------|-------|--------|---------------|
| 1     | 0         | 0     | False  | Cold start (learned weights) |
| 2     | 4         | 3     | True   | GP-driven (A/B/C/D) |
| 3     | 8         | 6     | True   | GP-driven (refined) |
| 4+    | 12+       | 9+    | True   | GP-driven (converging) |

---

## 🐛 Common Issues

### Issue 1: Identical Proposals
**Symptom:** All rounds have same weights
**Check:** `candidates: 0` in all rounds
**Cause:** Wrong endpoint or force_recreate
**Fix:** Use `/api/pbo/refine-next-round` for Round 2+

### Issue 2: GP Never Fitted
**Symptom:** `fitted: False` always
**Check:** Logs for `[PBO FIT]` message
**Cause:** Not enough candidates/duels
**Fix:** Verify candidates are being added

### Issue 3: State Resets
**Symptom:** Candidates/duels disappear
**Check:** Logs for `force_recreate=True`
**Fix:** Only use force_recreate during init

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `PBO_DEBUG_GUIDE.md` | Complete debugging guide |
| `PBO_MODES_VERIFICATION.md` | Verify both modes work |
| `PBO_DEBUG_SUMMARY.md` | Implementation summary |
| `PBO_DEBUG_QUICK_REF.md` | This quick reference |

---

## 🔗 Diagnostic Endpoint Response

```json
{
  "pbo_state": {
    "num_candidates": 4,
    "num_duels": 3,
    "fitted": true
  },
  "concept_weights": {
    "top_5": [...]
  },
  "recent_candidates": [...],
  "recent_duels": [...]
}
```

**Key Fields:**
- `num_candidates`: Should increase each round
- `num_duels`: Should be `num_candidates - 1` per round
- `fitted`: Should be `true` after Round 1

---

## 📞 Quick Test

```python
import requests

r = requests.get("http://localhost:8765/api/pbo/debug-state", 
                 params={"session_id": "<SESSION>", "stage": "impression"})
                 
state = r.json()['pbo_state']
print(f"Candidates: {state['num_candidates']}")
print(f"Duels: {state['num_duels']}")
print(f"Fitted: {state['fitted']}")
```

---

## ✅ Verification Checklist

- [ ] Server logs show correct endpoint sequence
- [ ] PBO state evolves (candidates/duels accumulate)
- [ ] GP fitted after Round 1
- [ ] Proposals different in Round 2+
- [ ] Diagnostic endpoint returns valid data
- [ ] Weight files show evolution

