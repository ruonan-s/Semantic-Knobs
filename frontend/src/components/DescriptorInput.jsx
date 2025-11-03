import React, { useState } from 'react';
export default function DescriptorInput({ onSubmit }) {
  const [text, setText] = useState('');
  return (
    <div className="mt-6">
      <textarea
        rows={3}
        className="w-full p-2 border rounded"
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Describe your environment..."
      />
      <button
        className="mt-2 px-4 py-2 rounded shadow bg-blue-500 text-white"
        onClick={() => onSubmit(text)}
        disabled={!text.trim()}
      >Start</button>
    </div>
  );
}