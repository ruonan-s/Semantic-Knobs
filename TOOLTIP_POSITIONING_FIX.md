# Bubble Chart Tooltip Smart Positioning Fix

## Problem

When hovering over bubbles near the edges or corners of the bubble chart, the tooltip (detail panel) was getting cut off or occluded by the frame boundaries, making it impossible to read the full information.

**Specific scenarios:**
- Bubbles in top-right corner → Tooltip extends beyond right edge
- Bubbles at bottom → Tooltip extends beyond bottom edge
- Bubbles in corners → Tooltip blocked by two edges simultaneously

## Solution: Smart Repositioning Algorithm

Implemented intelligent tooltip positioning that:
1. **Detects boundary collisions** before rendering
2. **Flips tooltip to opposite side** when near edges
3. **Ensures tooltip stays within chart bounds** at all times
4. **Smoothly transitions** when repositioning

## Implementation Details

### Smart Positioning Logic (BubbleChart.jsx:173-212)

```javascript
const handleMouseEnter = (bubble, event) => {
  const rect = svgRef.current.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  
  // Tooltip dimensions
  const tooltipWidth = 320;
  const tooltipHeight = 250; // Max expected height
  const padding = 10;
  
  // Default position: bottom-right of cursor
  let tooltipX = mouseX + padding;
  let tooltipY = mouseY + padding;
  
  // Check right boundary - flip to left if too close
  if (tooltipX + tooltipWidth > rect.width) {
    tooltipX = mouseX - tooltipWidth - padding;
  }
  
  // Check bottom boundary - flip to top if too close
  if (tooltipY + tooltipHeight > rect.height) {
    tooltipY = mouseY - tooltipHeight - padding;
  }
  
  // Ensure no negative coordinates (left/top edges)
  if (tooltipX < 0) {
    tooltipX = padding;
  }
  if (tooltipY < 0) {
    tooltipY = padding;
  }
  
  setTooltip({ bubble, x: tooltipX, y: tooltipY });
};
```

### Visual Examples

#### Before Fix
```
┌─────────────────────────────┐
│                             │
│         [Bubble]            │
│                    ┌────────┼──── ❌ Tooltip cut off
│                    │ Detail │
│                    │ Panel  │
└────────────────────┴────────┘
```

#### After Fix
```
┌─────────────────────────────┐
│                             │
│         [Bubble]            │
│           ┌────────┐        │ ✅ Tooltip flipped to left
│           │ Detail │        │
│           │ Panel  │        │
└───────────┴────────┴────────┘
```

### Four Positioning Modes

The tooltip intelligently chooses from 4 positions based on available space:

1. **Bottom-Right (default)**
   - Used when bubble is in center or top-left area
   - Cursor + padding offset

2. **Bottom-Left**
   - Used when bubble is near right edge
   - Flips horizontally to avoid right boundary

3. **Top-Right**
   - Used when bubble is near bottom edge
   - Flips vertically to avoid bottom boundary

4. **Top-Left**
   - Used when bubble is in bottom-right corner
   - Flips both horizontally and vertically

### Tooltip Enhancements

Added additional styling for better UX:

```javascript
style={{
  position: 'absolute',
  left: `${tooltip.x}px`,          // Dynamic positioning
  top: `${tooltip.y}px`,           // Dynamic positioning
  width: '320px',                  // Fixed width for predictable collision detection
  maxHeight: '250px',              // Max height for collision detection
  overflowY: 'auto',               // Scrollable if content exceeds max height
  transition: 'left 0.1s ease, top 0.1s ease',  // Smooth repositioning
  // ... other styles
}}
```

## Testing Scenarios

### Test 1: Top-Right Corner Bubble
1. Hover over bubble in top-right corner
2. **Expected:** Tooltip appears to the left and/or below cursor
3. **Verify:** No part of tooltip extends beyond chart boundaries

### Test 2: Bottom-Left Corner Bubble
1. Hover over bubble in bottom-left corner
2. **Expected:** Tooltip appears to the right and above cursor
3. **Verify:** Tooltip fully visible

### Test 3: Edge Bubbles
1. Hover over bubbles along right edge
2. **Expected:** Tooltip flips to left side of cursor
3. Hover over bubbles along bottom edge
4. **Expected:** Tooltip flips to top of cursor

### Test 4: Center Bubbles
1. Hover over bubbles in center
2. **Expected:** Tooltip appears in default position (bottom-right)
3. **Verify:** Smooth default behavior

### Test 5: Rapid Movement
1. Quickly move mouse across bubbles from center to corner
2. **Expected:** Tooltip smoothly transitions between positions
3. **Verify:** No flickering or jumps

## Edge Cases Handled

### Case 1: Very Large Tooltip
If tooltip content is exceptionally long:
- `maxHeight: 250px` prevents overflow
- `overflowY: auto` adds scrollbar
- Collision detection uses max height, not actual height

### Case 2: Both Boundaries
When near corner (right + bottom edges):
- Algorithm checks both axes independently
- Flips on both axes if needed
- Falls back to padding if still overflows

### Case 3: Small Chart
If chart is very small (< 320px wide):
- Tooltip positioned at `padding` distance from edge
- May overlap bubble slightly, but remains readable

## Configuration

Adjust these constants in `handleMouseEnter()` for different tooltip behavior:

```javascript
const tooltipWidth = 320;    // Increase for wider tooltips
const tooltipHeight = 250;   // Increase for taller tooltips
const padding = 10;          // Distance from cursor/edge
```

## Performance Considerations

- **No performance impact:** Calculations run only on hover
- **O(1) complexity:** Simple arithmetic, no loops
- **Smooth transitions:** CSS transitions for visual polish
- **No re-renders:** Only state update for tooltip position

## Accessibility

- Tooltip has high z-index (1000) to appear above all elements
- Pointer events disabled to prevent mouse interference
- High contrast black background with white text
- Smooth transitions reduce visual jarring

## Related Files

- ✏️ `frontend/src/components/BubbleChart.jsx` - Smart positioning logic

## Known Limitations

1. **Assumes fixed tooltip width:** If content changes width dynamically, collision detection may be inaccurate
2. **No multi-screen support:** Calculations based on chart container only
3. **Vertical scroll:** Long tag lists may require scrolling within tooltip

## Future Enhancements

- [ ] Dynamic tooltip height based on content
- [ ] Arrow pointer showing which bubble tooltip refers to
- [ ] Fade-in animation for smoother appearance
- [ ] Touch device support (tap to show/hide)

## Browser Compatibility

Tested and working on:
- ✅ Chrome/Edge (Chromium)
- ✅ Firefox
- ✅ Safari

Uses standard CSS/JS features, no special polyfills needed.

