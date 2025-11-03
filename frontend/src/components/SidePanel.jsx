import React from 'react';
export default function SidePanel({ tags, onClose }) {
  return (
    <div className="fixed top-0 right-0 w-64 h-full bg-white p-4 shadow-lg">
      <button onClick={onClose} className="mb-4">Close</button>
      <h3 className="font-semibold mb-2">Tags</h3>
      <ul className="list-disc list-inside">
        {tags.map(t => <li key={t}>{t}</li>)}
      </ul>
    </div>
  );
}