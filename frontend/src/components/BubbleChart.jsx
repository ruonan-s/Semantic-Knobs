import React, { useEffect, useRef, useState } from 'react';

/**
 * BubbleChart component for visualizing concept weights
 * Shows top 20 clusters OR all clusters with weights > 0
 */
function BubbleChart({ concepts, onConceptClick }) {
  const svgRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 500 });
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    if (!svgRef.current) return;

    // Update dimensions based on container
    const updateDimensions = () => {
      const container = svgRef.current.parentElement;
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: Math.max(500, container.clientHeight)
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Compute bubble positions and sizes with circle packing
  const bubbles = React.useMemo(() => {
    if (!concepts || concepts.length === 0) return [];

    const totalK = concepts.length;
    const w_base = 1.0 / totalK;
    const delta = 0.2 / totalK;

    const { width, height } = dimensions;
    const centerX = width / 2;
    const centerY = height / 2;

    // Sort concepts by weight (descending)
    const sortedConcepts = [...concepts].sort((a, b) => (b.state.ema_w || 0) - (a.state.ema_w || 0));
    
    // Filter: Show top 20 OR all with weight > 0
    const positiveWeightConcepts = sortedConcepts.filter(c => (c.state.ema_w || 0) > 0);
    const allConcepts = positiveWeightConcepts.length <= 20 
      ? positiveWeightConcepts 
      : sortedConcepts.slice(0, 20);
    
    // Minimal logging for performance
    console.log(`[BubbleChart] Rendered: ${allConcepts.length}/${totalK} concepts`);

    // Calculate bubble sizes based on ema_w
    const maxWeight = Math.max(...allConcepts.map(c => c.state.ema_w || 0));
    const minWeight = Math.min(...allConcepts.map(c => c.state.ema_w || 0));
    const weightRange = maxWeight - minWeight || 1;

    // Create bubbles with initial positions
    const initialBubbles = allConcepts.map((concept, index) => {
      const weight = concept.state.ema_w || 0;
      const normalizedWeight = (weight - minWeight) / weightRange;
      
      // Radius scaled by weight - adjusted for better text display
      // Use concept label length to influence size too
      const labelFactor = Math.min(1, concept.label.length / 15);
      const minRadius = 30 + labelFactor * 15; // Larger bubbles for longer names
      const maxRadius = 110;
      const radius = minRadius + Math.sqrt(normalizedWeight) * (maxRadius - minRadius);

      // Use gradient color based on weight (lighter = lower weight, darker = higher weight)
      // Removed color-coding by likes/dislikes - only size matters
      const normalizedForColor = Math.max(0, Math.min(1, normalizedWeight));
      
      // Gradient from light blue to deep blue
      const lightness = 75 - (normalizedForColor * 30); // 75% to 45%
      const saturation = 40 + (normalizedForColor * 30); // 40% to 70%
      const color = `hsl(210, ${saturation}%, ${lightness}%)`;
      
      // Status based on weight only
      let status;
      if (weight >= w_base + delta) {
        status = 'high-weight';
      } else if (weight <= w_base - delta) {
        status = 'low-weight';
      } else {
        status = 'medium-weight';
      }

      // Initial spiral positioning
      const angle = index * 2.4;
      const spiral = Math.sqrt(index + 1) * 15;
      
      return {
        id: concept.id,
        label: concept.label,
        x: centerX + Math.cos(angle) * spiral,
        y: centerY + Math.sin(angle) * spiral,
        radius,
        color,
        status,
        weight: weight,
        member_tags: concept.member_tags || [],
        state: concept.state,
        vx: 0,
        vy: 0
      };
    });

    // Simple physics simulation for circle packing
    const iterations = 50;
    const alpha = 0.3;
    const padding = 0; // No padding - bubbles touch each other
    
    for (let iter = 0; iter < iterations; iter++) {
      // Apply forces
      for (let i = 0; i < initialBubbles.length; i++) {
        const bubble = initialBubbles[i];
        
        // Center force (weak)
        const dx = centerX - bubble.x;
        const dy = centerY - bubble.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0) {
          bubble.vx += (dx / dist) * alpha * 0.01;
          bubble.vy += (dy / dist) * alpha * 0.01;
        }
        
        // Collision with other bubbles
        for (let j = i + 1; j < initialBubbles.length; j++) {
          const other = initialBubbles[j];
          const dx = other.x - bubble.x;
          const dy = other.y - bubble.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const minDist = bubble.radius + other.radius + padding;
          
          if (dist < minDist && dist > 0) {
            const force = (minDist - dist) / dist * alpha;
            const fx = dx * force;
            const fy = dy * force;
            
            bubble.vx -= fx;
            bubble.vy -= fy;
            other.vx += fx;
            other.vy += fy;
          }
        }
      }
      
      // Update positions
      for (const bubble of initialBubbles) {
        bubble.x += bubble.vx;
        bubble.y += bubble.vy;
        bubble.vx *= 0.9; // damping
        bubble.vy *= 0.9;
        
        // Keep within bounds
        bubble.x = Math.max(bubble.radius + 10, Math.min(width - bubble.radius - 10, bubble.x));
        bubble.y = Math.max(bubble.radius + 10, Math.min(height - bubble.radius - 10, bubble.y));
      }
    }

    return initialBubbles;
  }, [concepts, dimensions]);

  const handleBubbleClick = (bubble) => {
    if (onConceptClick) {
      onConceptClick(bubble);
    }
  };

  const handleMouseEnter = (bubble, event) => {
    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Tooltip dimensions (approximate)
    const tooltipWidth = 320;
    const tooltipHeight = 250; // Max expected height
    const padding = 10;
    
    // Calculate tooltip position with smart repositioning
    let tooltipX = mouseX + padding;
    let tooltipY = mouseY + padding;
    
    // Check right boundary - flip to left if too close to edge
    if (tooltipX + tooltipWidth > rect.width) {
      tooltipX = mouseX - tooltipWidth - padding;
    }
    
    // Check bottom boundary - flip to top if too close to edge
    if (tooltipY + tooltipHeight > rect.height) {
      tooltipY = mouseY - tooltipHeight - padding;
    }
    
    // Ensure tooltip doesn't go off left edge
    if (tooltipX < 0) {
      tooltipX = padding;
    }
    
    // Ensure tooltip doesn't go off top edge
    if (tooltipY < 0) {
      tooltipY = padding;
    }
    
    setTooltip({
      bubble,
      x: tooltipX,
      y: tooltipY
    });
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  return (
    <div style={{ 
      position: 'relative', 
      width: '100%', 
      height: '100%',
      minHeight: '500px',
      backgroundColor: '#ffffff',
      borderRadius: '8px',
      border: '1px solid #e0e0e0',
      overflow: 'hidden'
    }}>
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ display: 'block' }}
      >
        {/* Draw bubbles */}
        {bubbles.map((bubble) => {
          const weightPercent = (bubble.weight * 100).toFixed(1);
          const fontSize = Math.max(9, Math.min(14, bubble.radius / 3.5));
          
          // Split label into words for wrapping
          const words = bubble.label.split(' ');
          const lines = [];
          let currentLine = '';
          const maxCharsPerLine = Math.max(8, Math.floor(bubble.radius / 3.5));
          
          // Build lines by adding words
          for (const word of words) {
            if ((currentLine + ' ' + word).trim().length <= maxCharsPerLine) {
              currentLine = (currentLine + ' ' + word).trim();
            } else {
              if (currentLine) lines.push(currentLine);
              currentLine = word;
            }
          }
          if (currentLine) lines.push(currentLine);
          
          // Limit to 3 lines for very long labels
          const displayLines = lines.slice(0, 3);
          if (lines.length > 3) {
            displayLines[2] = displayLines[2].substring(0, maxCharsPerLine - 2) + '..';
          }
          
          const lineHeight = fontSize * 1.2;
          const totalTextHeight = displayLines.length * lineHeight + lineHeight * 0.8; // +percentage line
          const startY = bubble.y - totalTextHeight / 2 + lineHeight / 2;
          
          return (
            <g key={bubble.id}>
              {/* Bubble circle */}
              <circle
                cx={bubble.x}
                cy={bubble.y}
                r={bubble.radius}
                fill={bubble.color}
                opacity={0.85}
                stroke="#fff"
                strokeWidth={2}
                style={{
                  cursor: 'pointer',
                  filter: 'drop-shadow(0px 2px 4px rgba(0,0,0,0.2))',
                }}
                onClick={() => handleBubbleClick(bubble)}
                onMouseEnter={(e) => handleMouseEnter(bubble, e)}
                onMouseLeave={handleMouseLeave}
              />
              
              {/* Multi-line label and percentage inside bubble */}
              {bubble.radius > 25 && (
                <>
                  {displayLines.map((line, i) => (
                    <text
                      key={i}
                      x={bubble.x}
                      y={startY + i * lineHeight}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill="#fff"
                      fontSize={`${fontSize}px`}
                      fontWeight="700"
                      pointerEvents="none"
                      style={{
                        textShadow: '1px 1px 3px rgba(0,0,0,0.5)',
                        userSelect: 'none'
                      }}
                    >
                      {line}
                    </text>
                  ))}
                  <text
                    x={bubble.x}
                    y={startY + displayLines.length * lineHeight}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="#fff"
                    fontSize={`${fontSize * 0.85}px`}
                    fontWeight="600"
                    pointerEvents="none"
                    style={{
                      textShadow: '1px 1px 3px rgba(0,0,0,0.5)',
                      userSelect: 'none'
                    }}
                  >
                    {weightPercent}%
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>

      {/* Tooltip with smart positioning */}
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: `${tooltip.x}px`,
            top: `${tooltip.y}px`,
            backgroundColor: 'rgba(0, 0, 0, 0.95)',
            color: 'white',
            padding: '14px 16px',
            borderRadius: '8px',
            fontSize: '12px',
            maxWidth: '320px',
            width: '320px',
            maxHeight: '250px',
            overflowY: 'auto',
            zIndex: 1000,
            pointerEvents: 'none',
            boxShadow: '0 6px 20px rgba(0,0,0,0.5)',
            border: '1px solid rgba(255,255,255,0.2)',
            transition: 'left 0.1s ease, top 0.1s ease'
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '8px', fontSize: '15px', color: '#ffd700' }}>
            {tooltip.bubble.label}
          </div>
          
          <div style={{ marginBottom: '6px' }}>
            <span><strong>Weight:</strong> {(tooltip.bubble.weight * 100).toFixed(2)}%</span>
          </div>
          
          <div style={{ marginBottom: '8px', fontSize: '11px', opacity: 0.9 }}>
            👍 {tooltip.bubble.state.like_count} likes | 👎 {tooltip.bubble.state.dislike_count} dislikes
          </div>
          
          {tooltip.bubble.member_tags && tooltip.bubble.member_tags.length > 0 && (
            <div style={{ 
              marginTop: '10px', 
              paddingTop: '10px', 
              borderTop: '1px solid rgba(255,255,255,0.3)' 
            }}>
              <strong style={{ color: '#90caf9' }}>
                {tooltip.bubble.member_tags.length === 1 ? 'Tag:' : `Merged Tags (${tooltip.bubble.member_tags.length}):`}
              </strong>
              <div style={{ 
                marginTop: '6px',
                maxHeight: '150px',
                overflowY: 'auto'
              }}>
                {tooltip.bubble.member_tags.map((tag, idx) => (
                  <div 
                    key={idx} 
                    style={{ 
                      padding: '3px 0',
                      fontSize: '11px',
                      color: '#e0e0e0'
                    }}
                  >
                    • {tag}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        padding: '12px 14px',
        borderRadius: '8px',
        border: '1px solid #e0e0e0',
        fontSize: '12px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px', fontSize: '13px' }}>Bubble Key</div>
        
        {/* Color gradient explanation */}
        <div style={{ marginBottom: '10px' }}>
          <div style={{ fontSize: '11px', marginBottom: '4px', color: '#666' }}>Color Intensity:</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ 
              width: '30px', 
              height: '16px', 
              background: 'linear-gradient(to right, hsl(210, 40%, 75%), hsl(210, 70%, 45%))',
              borderRadius: '3px',
              border: '1px solid #ddd'
            }} />
            <span style={{ fontSize: '10px', color: '#666' }}>Low → High weight</span>
          </div>
        </div>
        
        {/* Size explanation */}
        <div style={{ marginBottom: '10px' }}>
          <div style={{ fontSize: '11px', marginBottom: '4px', color: '#666' }}>Bubble Size:</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ 
              width: '10px', 
              height: '10px', 
              backgroundColor: 'hsl(210, 55%, 60%)', 
              borderRadius: '50%',
              border: '1px solid #ddd'
            }} />
            <div style={{ 
              width: '16px', 
              height: '16px', 
              backgroundColor: 'hsl(210, 55%, 60%)', 
              borderRadius: '50%',
              border: '1px solid #ddd'
            }} />
            <span style={{ fontSize: '10px', color: '#666' }}>= Concept weight</span>
          </div>
        </div>
        
        <div style={{
          paddingTop: '8px',
          borderTop: '1px solid #e0e0e0',
          fontSize: '11px',
          color: '#666'
        }}>
          <div style={{ marginBottom: '2px' }}>
            <strong>Showing:</strong> {bubbles.length} of {concepts.length} concepts
          </div>
          <div style={{ fontSize: '10px', color: '#999', marginTop: '2px' }}>
            (Top 20 or all weight {'>'} 0)
          </div>
          <div style={{ fontSize: '10px', fontStyle: 'italic', color: '#999', marginTop: '4px' }}>
            Click bubbles to see details
          </div>
        </div>
      </div>
    </div>
  );
}

export default BubbleChart;

