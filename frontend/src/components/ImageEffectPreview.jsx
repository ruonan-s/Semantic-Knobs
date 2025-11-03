import React from 'react';

/**
 * ImageEffectPreview component shows how each image is affected by concept weights
 * Displays as horizontal bars showing predicted effect scores
 */
function ImageEffectPreview({ images, imageEffects, selectedImage, onImageClick }) {
  // Find min and max effects for scaling
  const effects = Object.values(imageEffects || {});
  const maxEffect = effects.length > 0 ? Math.max(...effects, 0) : 1;
  const minEffect = effects.length > 0 ? Math.min(...effects, 0) : 0;
  const effectRange = Math.max(Math.abs(maxEffect), Math.abs(minEffect));

  return (
    <div style={{
      backgroundColor: '#f9f9f9',
      borderRadius: '8px',
      border: '1px solid #e0e0e0',
      padding: '16px'
    }}>
      <h3 style={{
        margin: '0 0 16px 0',
        fontSize: '16px',
        fontWeight: '600',
        color: '#333'
      }}>
        Image Preference Preview
      </h3>
      <p style={{
        margin: '0 0 16px 0',
        fontSize: '13px',
        color: '#666'
      }}>
        These bars show how well each image aligns with your current preferences. 
        Higher values indicate better alignment.
      </p>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        {images.map((image) => {
          const effect = imageEffects[image.id] || 0;
          const isSelected = selectedImage === image.id;
          
          // Calculate bar width (0-100%)
          const barWidth = effectRange > 0 
            ? Math.abs(effect) / effectRange * 100 
            : 0;
          
          const isPositive = effect >= 0;
          const barColor = isPositive ? '#4CAF50' : '#f44336';

          return (
            <div
              key={image.id}
              onClick={() => onImageClick && onImageClick(image.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px',
                backgroundColor: isSelected ? '#e3f2fd' : 'white',
                borderRadius: '6px',
                border: isSelected ? '2px solid #2196F3' : '1px solid #e0e0e0',
                cursor: onImageClick ? 'pointer' : 'default',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = '#f5f5f5';
                }
              }}
              onMouseLeave={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = 'white';
                }
              }}
            >
              {/* Thumbnail */}
              <img
                src={image.url}
                alt={image.id}
                style={{
                  width: '60px',
                  height: '60px',
                  objectFit: 'cover',
                  borderRadius: '4px',
                  border: '1px solid #ddd'
                }}
              />

              {/* Bar chart */}
              <div style={{ flex: 1 }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '6px'
                }}>
                  <span style={{
                    fontSize: '13px',
                    fontWeight: '500',
                    color: '#333'
                  }}>
                    {image.id}
                  </span>
                  <span style={{
                    fontSize: '13px',
                    fontWeight: 'bold',
                    color: barColor
                  }}>
                    {effect >= 0 ? '+' : ''}{effect.toFixed(3)}
                  </span>
                </div>

                {/* Progress bar background */}
                <div style={{
                  position: 'relative',
                  width: '100%',
                  height: '24px',
                  backgroundColor: '#e0e0e0',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  {/* Center line for zero */}
                  <div style={{
                    position: 'absolute',
                    left: '50%',
                    top: 0,
                    bottom: 0,
                    width: '2px',
                    backgroundColor: '#999',
                    zIndex: 1
                  }} />

                  {/* Effect bar */}
                  {effect !== 0 && (
                    <div style={{
                      position: 'absolute',
                      left: isPositive ? '50%' : `${50 - barWidth / 2}%`,
                      top: 0,
                      bottom: 0,
                      width: `${barWidth / 2}%`,
                      backgroundColor: barColor,
                      opacity: 0.8,
                      transition: 'all 0.3s ease'
                    }} />
                  )}
                </div>
              </div>

              {/* Selected indicator */}
              {isSelected && (
                <div style={{
                  fontSize: '20px',
                  color: '#2196F3'
                }}>
                  ✓
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Scale legend */}
      <div style={{
        marginTop: '16px',
        padding: '12px',
        backgroundColor: 'white',
        borderRadius: '6px',
        border: '1px solid #e0e0e0',
        fontSize: '12px',
        color: '#666'
      }}>
        <div style={{ fontWeight: '600', marginBottom: '8px' }}>How to Read:</div>
        <div style={{ marginBottom: '4px' }}>
          • <span style={{ color: '#4CAF50', fontWeight: '600' }}>Green bars</span> = Strong positive alignment with your preferences
        </div>
        <div style={{ marginBottom: '4px' }}>
          • <span style={{ color: '#f44336', fontWeight: '600' }}>Red bars</span> = Contains concepts you dislike
        </div>
        <div>
          • Longer bars = Stronger effect (positive or negative)
        </div>
      </div>
    </div>
  );
}

export default ImageEffectPreview;

