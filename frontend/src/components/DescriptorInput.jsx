import React, { useState } from 'react';

export default function DescriptorInput({ onSubmit }) {
  const [adjective, setAdjective] = useState('');
  const [location, setLocation] = useState('');
  
  const handleSubmit = () => {
    // Concatenate adjective and location to form the complete descriptor
    const descriptor = `${adjective.trim()} ${location.trim()}`;
    onSubmit({ adjective: adjective.trim(), location: location.trim(), descriptor });
  };
  
  const isValid = adjective.trim() && location.trim();
  
  return (
    <div className="mt-6">
      <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
        <div style={{ flex: 1 }}>
          <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500', color: '#374151' }}>
            Adjective
          </label>
          <input
            type="text"
            className="w-full p-2 border rounded"
            value={adjective}
            onChange={e => setAdjective(e.target.value)}
            placeholder="e.g., cozy, modern, rustic..."
            style={{ width: '100%' }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500', color: '#374151' }}>
            Location
          </label>
          <input
            type="text"
            className="w-full p-2 border rounded"
            value={location}
            onChange={e => setLocation(e.target.value)}
            placeholder="e.g., space, bedroom, kitchen..."
            style={{ width: '100%' }}
          />
        </div>
      </div>
      <button
        className="mt-2 px-4 py-2 rounded shadow bg-blue-500 text-white"
        onClick={handleSubmit}
        disabled={!isValid}
      >Start</button>
    </div>
  );
}
