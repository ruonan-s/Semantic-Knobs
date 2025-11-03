import React, { useState } from 'react';

/**
 * ConceptLists component for displaying and reordering positive/negative/neutral concepts
 * Supports drag-and-drop reordering within positive and negative lists
 */
function ConceptLists({ concepts, categorized, onRankingChange }) {
  const [draggedItem, setDraggedItem] = useState(null);
  const [draggedFrom, setDraggedFrom] = useState(null);

  // Get concept by id
  const getConceptById = (id) => {
    return concepts.find(c => c.id === id);
  };

  const handleDragStart = (e, conceptId, listType) => {
    setDraggedItem(conceptId);
    setDraggedFrom(listType);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', e.target);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    return false;
  };

  const handleDrop = (e, targetId, targetList) => {
    e.preventDefault();
    e.stopPropagation();

    console.log('[DRAG DROP] Drop event:', {
      draggedItem,
      draggedFrom,
      targetId,
      targetList,
      crossList: draggedFrom !== targetList
    });

    if (!draggedItem) {
      console.log('[DRAG DROP] Rejected: no dragged item');
      setDraggedItem(null);
      setDraggedFrom(null);
      return;
    }

    // Allow dragging between all lists including neutral

    // Build new lists based on drag operation
    let newPositive = [...categorized.positive];
    let newNegative = [...categorized.negative];
    let newNeutral = [...categorized.neutral];

    // Case 1: Same list - reordering (for positive/negative/neutral)
    if (draggedFrom === targetList) {
      let sourceList;
      if (targetList === 'positive') {
        sourceList = newPositive;
      } else if (targetList === 'negative') {
        sourceList = newNegative;
      } else {
        sourceList = newNeutral;
      }

      const draggedIndex = sourceList.indexOf(draggedItem);
      const targetIndex = sourceList.indexOf(targetId);

      console.log('[DRAG DROP] Same-list reorder:', {
        list: targetList,
        draggedIndex,
        targetIndex
      });

      if (draggedIndex === targetIndex) {
        setDraggedItem(null);
        setDraggedFrom(null);
        return;
      }

      // Remove from old position and insert at new position
      sourceList.splice(draggedIndex, 1);
      sourceList.splice(targetIndex, 0, draggedItem);

      if (targetList === 'positive') {
        newPositive = sourceList;
      } else if (targetList === 'negative') {
        newNegative = sourceList;
      } else {
        newNeutral = sourceList;
      }
    }
    // Case 2: Cross-list - move between lists (including to/from neutral)
    else {
      console.log('[DRAG DROP] Cross-list move:', {
        from: draggedFrom,
        to: targetList
      });

      // Remove from source list
      if (draggedFrom === 'positive') {
        newPositive = newPositive.filter(id => id !== draggedItem);
      } else if (draggedFrom === 'negative') {
        newNegative = newNegative.filter(id => id !== draggedItem);
      } else if (draggedFrom === 'neutral') {
        newNeutral = newNeutral.filter(id => id !== draggedItem);
      }

      // Add to target list at the position of targetId
      if (targetList === 'positive') {
        const targetIndex = newPositive.indexOf(targetId);
        newPositive.splice(targetIndex, 0, draggedItem);
      } else if (targetList === 'negative') {
        const targetIndex = newNegative.indexOf(targetId);
        newNegative.splice(targetIndex, 0, draggedItem);
      } else if (targetList === 'neutral') {
        const targetIndex = newNeutral.indexOf(targetId);
        newNeutral.splice(targetIndex, 0, draggedItem);
      }
    }

    console.log('[DRAG DROP] New lists:', {
      positive: newPositive,
      negative: newNegative,
      neutral: newNeutral
    });

    // Update the rankings with explicit lists
    onRankingChange(newPositive, newNegative);

    setDraggedItem(null);
    setDraggedFrom(null);
  };

  const renderConceptItem = (conceptId, listType, index) => {
    const concept = getConceptById(conceptId);
    if (!concept) return null;

    const isDragging = draggedItem === conceptId;
    // Show drag target highlight for all lists when something is being dragged
    const isDragTarget = draggedItem && draggedItem !== conceptId;
    const weight = (concept.state.w * 100).toFixed(1);

    return (
      <div
        key={conceptId}
        draggable={true}
        onDragStart={(e) => handleDragStart(e, conceptId, listType)}
        onDragOver={handleDragOver}
        onDrop={(e) => handleDrop(e, conceptId, listType)}
        style={{
          padding: '10px 12px',
          marginBottom: '8px',
          backgroundColor: isDragging ? '#f0f0f0' : isDragTarget ? '#e3f2fd' : 'white',
          border: isDragTarget ? '2px dashed #2196F3' : '1px solid #e0e0e0',
          borderRadius: '6px',
          cursor: 'move',
          opacity: isDragging ? 0.5 : 1,
          transition: 'all 0.2s ease',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <div style={{ flex: 1 }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center',
            marginBottom: '4px'
          }}>
            {listType !== 'neutral' && (
              <span style={{ 
                marginRight: '8px', 
                color: '#999',
                fontSize: '12px',
                fontWeight: 'bold'
              }}>
                #{index + 1}
              </span>
            )}
            <span style={{ fontWeight: '500', fontSize: '14px' }}>
              {concept.label}
            </span>
          </div>
          {concept.member_tags && concept.member_tags.length > 0 && (
            <div style={{ 
              fontSize: '11px', 
              color: '#555',
              marginTop: '6px',
              lineHeight: '1.4'
            }}>
              <span style={{ 
                fontSize: '10px', 
                fontWeight: '600', 
                color: '#888',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                {concept.member_tags.length === 1 ? 'Tag:' : `${concept.member_tags.length} tags:`}
              </span>
              <div style={{ marginTop: '3px' }}>
                {concept.member_tags.slice(0, 5).map((tag, idx) => (
                  <span 
                    key={idx}
                    style={{
                      display: 'inline-block',
                      marginRight: '4px',
                      marginBottom: '3px',
                      padding: '2px 6px',
                      backgroundColor: '#f0f0f0',
                      borderRadius: '3px',
                      fontSize: '10px',
                      color: '#444'
                    }}
                  >
                    {tag}
                  </span>
                ))}
                {concept.member_tags.length > 5 && (
                  <span style={{ 
                    fontSize: '10px', 
                    color: '#999',
                    fontStyle: 'italic'
                  }}>
                    +{concept.member_tags.length - 5} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          marginLeft: '12px'
        }}>
          <div style={{ 
            fontSize: '14px', 
            fontWeight: 'bold',
            color: listType === 'positive' ? '#4CAF50' : listType === 'negative' ? '#f44336' : '#999'
          }}>
            {weight}%
          </div>
          <div style={{ fontSize: '10px', color: '#999', marginTop: '2px' }}>
            👍 {concept.state.like_count} 👎 {concept.state.dislike_count}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ 
      display: 'flex', 
      gap: '20px',
      width: '100%',
      height: '100%'
    }}>
      {/* Positive Column */}
      <div style={{ 
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minWidth: '0'
      }}>
        <div style={{
          backgroundColor: '#e8f5e9',
          padding: '12px',
          borderRadius: '8px 8px 0 0',
          border: '1px solid #c8e6c9',
          borderBottom: 'none',
          fontWeight: '600',
          fontSize: '14px',
          color: '#2e7d32',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <span>✓ Positive Concepts</span>
            <div style={{ fontSize: '10px', fontWeight: 'normal', marginTop: '2px', opacity: 0.8 }}>
              Drag to reorder or move between lists
            </div>
          </div>
          <span style={{ 
            fontSize: '12px', 
            fontWeight: 'normal',
            backgroundColor: 'white',
            padding: '2px 8px',
            borderRadius: '12px'
          }}>
            {categorized.positive.length}
          </span>
        </div>
        <div style={{
          flex: 1,
          overflowY: 'auto',
          backgroundColor: '#f9f9f9',
          padding: '12px',
          borderRadius: '0 0 8px 8px',
          border: '1px solid #c8e6c9',
          borderTop: 'none',
          minHeight: '200px',
          maxHeight: '400px'
        }}>
          {categorized.positive.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              color: '#999', 
              padding: '20px',
              fontSize: '13px'
            }}>
              No positive concepts yet.<br/>
              Click tags to increase their importance.
            </div>
          ) : (
            categorized.positive.map((id, index) => renderConceptItem(id, 'positive', index))
          )}
        </div>
      </div>

      {/* Neutral Column */}
      <div style={{ 
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minWidth: '0'
      }}>
        <div style={{
          backgroundColor: '#f5f5f5',
          padding: '12px',
          borderRadius: '8px 8px 0 0',
          border: '1px solid #e0e0e0',
          borderBottom: 'none',
          fontWeight: '600',
          fontSize: '14px',
          color: '#666',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <span>— Neutral Concepts</span>
            <div style={{ fontSize: '10px', fontWeight: 'normal', marginTop: '2px', opacity: 0.8 }}>
              Drag to reorder or move between lists
            </div>
          </div>
          <span style={{ 
            fontSize: '12px', 
            fontWeight: 'normal',
            backgroundColor: 'white',
            padding: '2px 8px',
            borderRadius: '12px'
          }}>
            {categorized.neutral.length}
          </span>
        </div>
        <div style={{
          flex: 1,
          overflowY: 'auto',
          backgroundColor: '#f9f9f9',
          padding: '12px',
          borderRadius: '0 0 8px 8px',
          border: '1px solid #e0e0e0',
          borderTop: 'none',
          minHeight: '200px',
          maxHeight: '400px'
        }}>
          {categorized.neutral.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              color: '#999', 
              padding: '20px',
              fontSize: '13px'
            }}>
              No neutral concepts.
            </div>
          ) : (
            categorized.neutral.map((id, index) => renderConceptItem(id, 'neutral', index))
          )}
        </div>
      </div>

      {/* Negative Column */}
      <div style={{ 
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minWidth: '0'
      }}>
        <div style={{
          backgroundColor: '#ffebee',
          padding: '12px',
          borderRadius: '8px 8px 0 0',
          border: '1px solid #ffcdd2',
          borderBottom: 'none',
          fontWeight: '600',
          fontSize: '14px',
          color: '#c62828',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <span>✗ Negative Concepts</span>
            <div style={{ fontSize: '10px', fontWeight: 'normal', marginTop: '2px', opacity: 0.8 }}>
              Drag to reorder or move between lists
            </div>
          </div>
          <span style={{ 
            fontSize: '12px', 
            fontWeight: 'normal',
            backgroundColor: 'white',
            padding: '2px 8px',
            borderRadius: '12px'
          }}>
            {categorized.negative.length}
          </span>
        </div>
        <div style={{
          flex: 1,
          overflowY: 'auto',
          backgroundColor: '#f9f9f9',
          padding: '12px',
          borderRadius: '0 0 8px 8px',
          border: '1px solid #ffcdd2',
          borderTop: 'none',
          minHeight: '200px',
          maxHeight: '400px'
        }}>
          {categorized.negative.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              color: '#999', 
              padding: '20px',
              fontSize: '13px'
            }}>
              No negative concepts yet.<br/>
              Dislike tags to reduce their importance.
            </div>
          ) : (
            categorized.negative.map((id, index) => renderConceptItem(id, 'negative', index))
          )}
        </div>
      </div>
    </div>
  );
}

export default ConceptLists;

