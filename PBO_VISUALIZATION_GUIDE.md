# PBO Weight Evolution Visualization Guide

## 📊 **Interactive Visualization Tool**

This guide explains how to use the `visualize_pbo_weights.py` script to create interactive visualizations of concept weight evolution across PBO refinement rounds.

---

## 🚀 **Quick Start**

### 1. **Install Dependencies**

```bash
pip install plotly numpy
```

### 2. **Run the Visualization**

```bash
cd backend
python visualize_pbo_weights.py <session_path>
```

**Example:**
```bash
python visualize_pbo_weights.py sessions/[fast]_A_refreshing_space_by_the_sea_2025-11-07_02-33-39/impression_refinement/
```

### 3. **Provide Selections**

When prompted, enter which image (0-3) you selected each round:
- Press **Enter** to use default selections from logs
- Or type **0**, **1**, **2**, or **3** to specify manually

---

## 📈 **What the Visualization Shows**

### **Blue Dashed Lines: Average Weights**
- Average concept weight across all 4 proposals in each round
- Shows the "center of mass" of the PBO's exploration

### **Red Solid Lines: Selected Weights**
- Concept weights of the image you actually selected each round
- Shows your actual preference trajectory

### **Top 10 Concepts**
- Automatically identifies the 10 most important concepts (by average weight)
- Each concept gets 2 lines: average (blue) and selected (red)

---

## 🎯 **Interpreting the Results**

### ✅ **Good Learning Indicators**

**1. Selected weights diverge from average:**
```
Average (blue):  ─ ─ ─ ─ ─ ─ ─
Selected (red):  ━━━━━━━━━━━━━━
                          ↑
                    Different path!
```
**Means:** PBO is learning your specific preferences, not just averaging

**2. Selected weights show clear trends:**
```
Round 1 → 2 → 3 → 4 → 5
  0.2 → 0.3 → 0.5 → 0.6 → 0.8  ← Increasing (you like this concept!)
  0.4 → 0.3 → 0.2 → 0.1 → 0.05 ← Decreasing (you dislike this concept!)
```
**Means:** PBO is converging on your preferences

**3. Exploration then exploitation:**
```
Rounds 1-3: Wide spread, trying different concepts
Rounds 4-8: Narrowing focus, refining top concepts
Rounds 9-12: Converged, mostly exploiting best
```
**Means:** Healthy exploration/exploitation balance

---

### ⚠️ **Potential Issues**

**1. Selected = Average (always):**
```
Average (blue):  ─ ─ ─ ─ ─ ─ ─
Selected (red):  ━━━━━━━━━━━━━━
                 ↑
            Overlapping!
```
**Means:** PBO might not be learning (stuck in cold start)

**2. No clear trends:**
```
Round 1 → 2 → 3 → 4 → 5
  0.2 → 0.5 → 0.3 → 0.6 → 0.2  ← Random jumps!
```
**Means:** User preferences inconsistent OR GP not fitting properly

**3. All concepts decline:**
```
All weights decreasing across rounds
```
**Means:** Possible normalization issue or too many concepts

---

## 📊 **Example Analysis**

### **Your Current Session Results:**

**Top 10 Concepts (by average weight):**
1. `colorful eclectic decor`: 0.289
2. `coastal retreat location`: 0.188
3. `mediterranean aesthetic`: 0.099
4. `inviting serene vibe`: 0.085
5. `industrial loft style`: 0.080
6. `coastal minimalism`: 0.052
7. `open airy ambiance`: 0.036
8. `natural wood elements`: 0.023
9. `natural light enhances serenity`: 0.022
10. `plants enhance tranquility`: 0.022

### **What to Look For:**

**Concept 1: "colorful eclectic decor"**
- Highest average weight (0.289)
- Check if selected (red) is higher or lower than average (blue)
- **Higher:** You prefer MORE color than average
- **Lower:** You prefer LESS color than average

**Concept 2: "coastal retreat location"**
- Second highest (0.188)
- If selected line is increasing over rounds → you're learning to prefer this
- If selected line is decreasing → you're learning to avoid this

**Concepts 6-10: Lower weights**
- These are "minor" concepts
- Large swings in selected (red) line = you're sensitive to these details
- Flat selected line = you don't care much about these

---

## 🔍 **Interactive Features (Plotly HTML)**

When you open `pbo_weight_evolution.html` in a browser:

### **Hover:**
- Hover over any point to see exact values
- Shows: Round number, concept name, weight value

### **Click Legend:**
- Click a concept name in the legend to hide/show it
- Double-click to isolate just that concept

### **Zoom:**
- Click and drag to zoom into a region
- Double-click to reset zoom

### **Pan:**
- Drag to pan around the plot

### **Download:**
- Click camera icon to save as PNG

---

## 🛠️ **Customization**

### **Change Number of Top Concepts**

Edit `visualize_pbo_weights.py`:
```python
# Line 267
top_k = 10  # Change to 5, 15, 20, etc.
```

### **Change Colors**

```python
# Average lines (blue)
line=dict(dash='dash', width=2, color='blue')

# Selected lines (red)
line=dict(color='red', width=3)
```

### **Filter Specific Concepts**

Instead of automatic top-K, manually specify concepts:
```python
# Replace calculate_top_concepts() call with:
top_concepts = [0, 5, 8, 15]  # Concept indices you want
```

---

## 📁 **Output Files**

### **With Plotly (default):**
```
<session_path>/pbo_weight_evolution.html
```
- Interactive HTML visualization
- Open in any web browser
- 4-5 MB file size

### **Without Plotly (fallback):**
```
<session_path>/pbo_weight_evolution.png
```
- Static PNG image
- 1-2 MB file size
- Less interactive but works without dependencies

---

## 🧪 **Troubleshooting**

### **"No round directories found"**
**Problem:** Script can't find round_1, round_2, etc.
**Solution:** Check path is correct and contains round_X directories

### **"Plotly not found"**
**Problem:** Plotly not installed
**Solution:** `pip install plotly` or use matplotlib fallback

### **"Invalid input"**
**Problem:** Entered selection outside 0-3 range
**Solution:** Script will use default (0), or re-run and enter valid number

### **Lines are overlapping/hard to see**
**Problem:** Too many concepts displayed
**Solution:** Reduce `top_k` to 5-7 concepts

### **Red and blue lines identical**
**Problem:** You always selected the average proposal
**Solution:** This is rare but possible if proposal 0 is always closest to average

---

## 💡 **Tips for Best Results**

1. **Run after each session** to see learning progression in real-time
2. **Compare multiple sessions** to see if learning improves over time
3. **Look for convergence** - weights should stabilize after 5-10 rounds
4. **Identify surprises** - concepts you didn't expect to be important
5. **Validate with images** - do high-weight concepts match what you see?

---

## 🎨 **Advanced: Create Custom Views**

### **Individual Concept Panels**

For a cleaner view, plot each concept in its own subplot:

```python
from plotly.subplots import make_subplots

# Create 5x2 grid for top 10 concepts
fig = make_subplots(rows=5, cols=2, subplot_titles=[labels[i] for i in top_concepts])

for i, concept_idx in enumerate(top_concepts):
    row = i // 2 + 1
    col = i % 2 + 1
    
    # Add average line
    fig.add_trace(go.Scatter(...), row=row, col=col)
    
    # Add selected line
    fig.add_trace(go.Scatter(...), row=row, col=col)
```

### **Heatmap View**

Show all concepts x rounds as a heatmap:

```python
import plotly.express as px

# Create matrix: rows = concepts, cols = rounds
data = np.array([selected_weights[r] for r in range(num_rounds)]).T

fig = px.imshow(
    data,
    labels=dict(x="Round", y="Concept", color="Weight"),
    y=[labels[i] for i in range(len(labels))],
    aspect="auto",
    color_continuous_scale="RdBu_r"
)
```

---

## 📝 **Summary**

**The visualization helps you:**
- ✅ Verify PBO is learning your preferences
- ✅ Identify which concepts you care most about
- ✅ See exploration → exploitation transition
- ✅ Debug issues (stuck in cold start, inconsistent preferences, etc.)
- ✅ Understand your own aesthetic preferences quantitatively

**Use it to:**
- Validate the PBO system is working correctly
- Gain insights into your design preferences
- Debug and improve the PBO algorithm
- Create reports and presentations about the system

---

## 🚀 **Next Steps**

1. **Run the visualization** on your current session
2. **Open the HTML file** and explore interactively
3. **Look for learning indicators** (divergence, trends, convergence)
4. **Iterate:** If learning looks poor, adjust PBO hyperparameters
5. **Compare:** Run multiple sessions and compare learning curves

Happy visualizing! 📊✨

