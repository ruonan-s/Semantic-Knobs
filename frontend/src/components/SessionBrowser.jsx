import React, { useState, useEffect } from 'react';

/**
 * SessionBrowser component for selecting existing sessions
 * Used in Test Stage Refinement workflow
 */
function SessionBrowser({ onSessionSelect }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSession, setSelectedSession] = useState(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/list-sessions');
      
      if (!response.ok) {
        throw new Error(`Failed to load sessions: ${response.status}`);
      }
      
      const data = await response.json();
      setSessions(data.sessions || []);
      setError(null);
    } catch (err) {
      console.error('Error loading sessions:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSessionClick = (session) => {
    setSelectedSession(session.path);
  };

  const handleConfirmSelection = () => {
    if (selectedSession) {
      onSessionSelect(selectedSession);
    }
  };

  const filteredSessions = sessions.filter(session =>
    session.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: '#666'
      }}>
        <div style={{ fontSize: '18px', marginBottom: '10px' }}>Loading sessions...</div>
        <div style={{ fontSize: '14px' }}>Please wait</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center'
      }}>
        <div style={{ color: '#f44336', fontSize: '18px', marginBottom: '20px' }}>
          Error loading sessions
        </div>
        <div style={{ color: '#666', marginBottom: '20px' }}>
          {error}
        </div>
        <button
          onClick={loadSessions}
          style={{
            padding: '10px 20px',
            backgroundColor: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div style={{
      padding: '30px',
      maxWidth: '1000px',
      margin: '0 auto'
    }}>
      <div style={{
        marginBottom: '30px',
        textAlign: 'center'
      }}>
        <h2 style={{
          fontSize: '24px',
          fontWeight: '600',
          color: '#333',
          marginBottom: '10px'
        }}>
          Select Existing Session
        </h2>
        <p style={{
          color: '#666',
          fontSize: '14px'
        }}>
          Choose a session to resume and refine from any stage
        </p>
      </div>

      {/* Search Bar */}
      <div style={{
        marginBottom: '20px'
      }}>
        <input
          type="text"
          placeholder="Search sessions..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            width: '100%',
            padding: '12px 16px',
            fontSize: '14px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            outline: 'none',
            transition: 'border-color 0.2s'
          }}
          onFocus={(e) => e.target.style.borderColor = '#2196F3'}
          onBlur={(e) => e.target.style.borderColor = '#ddd'}
        />
      </div>

      {/* Sessions List */}
      <div style={{
        marginBottom: '20px',
        maxHeight: '500px',
        overflowY: 'auto',
        border: '1px solid #e0e0e0',
        borderRadius: '8px',
        backgroundColor: '#fafafa'
      }}>
        {filteredSessions.length === 0 ? (
          <div style={{
            padding: '40px',
            textAlign: 'center',
            color: '#999'
          }}>
            {searchTerm ? 'No sessions match your search' : 'No sessions found'}
          </div>
        ) : (
          filteredSessions.map((session) => (
            <div
              key={session.path}
              onClick={() => handleSessionClick(session)}
              style={{
                padding: '16px 20px',
                borderBottom: '1px solid #e0e0e0',
                cursor: 'pointer',
                backgroundColor: selectedSession === session.path ? '#e3f2fd' : 'white',
                transition: 'all 0.2s',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
              onMouseEnter={(e) => {
                if (selectedSession !== session.path) {
                  e.currentTarget.style.backgroundColor = '#f5f5f5';
                }
              }}
              onMouseLeave={(e) => {
                if (selectedSession !== session.path) {
                  e.currentTarget.style.backgroundColor = 'white';
                }
              }}
            >
              <div style={{ flex: 1 }}>
                <div style={{
                  fontSize: '15px',
                  fontWeight: '500',
                  color: '#333',
                  marginBottom: '6px'
                }}>
                  {session.name}
                </div>
                <div style={{
                  fontSize: '12px',
                  color: '#666'
                }}>
                  <span style={{ marginRight: '15px' }}>
                    📅 {session.timestamp}
                  </span>
                  <span>
                    ✓ {session.stage_count} stage{session.stage_count !== 1 ? 's' : ''} completed
                  </span>
                </div>
              </div>
              
              {selectedSession === session.path && (
                <div style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: '#2196F3',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '14px',
                  fontWeight: 'bold'
                }}>
                  ✓
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Action Buttons */}
      <div style={{
        display: 'flex',
        gap: '12px',
        justifyContent: 'flex-end'
      }}>
        <button
          onClick={() => window.location.reload()}
          style={{
            padding: '12px 24px',
            backgroundColor: '#f5f5f5',
            color: '#666',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#e0e0e0'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#f5f5f5'}
        >
          Cancel
        </button>
        
        <button
          onClick={handleConfirmSelection}
          disabled={!selectedSession}
          style={{
            padding: '12px 24px',
            backgroundColor: selectedSession ? '#2196F3' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: selectedSession ? 'pointer' : 'not-allowed',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            if (selectedSession) {
              e.target.style.backgroundColor = '#1976D2';
            }
          }}
          onMouseLeave={(e) => {
            if (selectedSession) {
              e.target.style.backgroundColor = '#2196F3';
            }
          }}
        >
          Continue →
        </button>
      </div>

      {/* Info Box */}
      <div style={{
        marginTop: '30px',
        padding: '16px',
        backgroundColor: '#f0f7ff',
        border: '1px solid #bbdefb',
        borderRadius: '8px',
        fontSize: '13px',
        color: '#555'
      }}>
        <strong style={{ color: '#1976D2' }}>💡 Tip:</strong> Select a session to load its existing images and tags. 
        You can then manipulate concepts and generate refinement images without starting from scratch.
      </div>
    </div>
  );
}

export default SessionBrowser;

