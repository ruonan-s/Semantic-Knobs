import React from 'react';
import { FiMoreVertical } from 'react-icons/fi';

export default function ImageGrid({ images, selected, onSelect, onTagClick }) {
  return (
    <div className="grid grid-cols-2 gap-4 mt-4">
      {images.map(img => (
        <div key={img.id} className="relative">
          <img
            src={img.url}
            alt={img.id}
            className={`w-full h-auto cursor-pointer rounded shadow ${selected === img.id ? 'ring-4 ring-blue-400' : ''}`}
            onClick={() => onSelect(img.id)}
          />
          <button
            className="absolute top-2 right-2 bg-white p-1 rounded-full"
            onClick={() => onTagClick(img)}
          >
            <FiMoreVertical />
          </button>
        </div>
      ))}
    </div>
  );
}