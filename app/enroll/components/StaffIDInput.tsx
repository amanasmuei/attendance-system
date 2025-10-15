'use client';

/**
 * Staff ID Input Component
 * Validates and submits staff ID for enrollment
 */

import React, { useState } from 'react';

export interface StaffIDInputProps {
  onSubmit: (staffId: string) => void;
  isLoading?: boolean;
}

export const StaffIDInput: React.FC<StaffIDInputProps> = ({ onSubmit, isLoading = false }) => {
  const [staffId, setStaffId] = useState('');
  const [error, setError] = useState<string | null>(null);

  const validateStaffId = (id: string): boolean => {
    // Basic validation - customize based on your requirements
    if (!id.trim()) {
      setError('Staff ID is required');
      return false;
    }

    if (id.length < 3) {
      setError('Staff ID must be at least 3 characters');
      return false;
    }

    // Example format: EMP12345, STF0001, etc.
    const pattern = /^[A-Z]{3}\d{4,}$/i;
    if (!pattern.test(id.trim())) {
      setError('Invalid format. Expected format: ABC1234 (3 letters + numbers)');
      return false;
    }

    return true;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const trimmedId = staffId.trim().toUpperCase();

    if (validateStaffId(trimmedId)) {
      onSubmit(trimmedId);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setStaffId(e.target.value);
    setError(null);
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="staffId" className="block text-sm font-medium text-gray-700 mb-2">
            Enter Your Staff ID
          </label>
          <input
            type="text"
            id="staffId"
            value={staffId}
            onChange={handleChange}
            placeholder="e.g., EMP12345"
            disabled={isLoading}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed uppercase"
            autoFocus
          />
          {error && (
            <p className="mt-2 text-sm text-red-600" role="alert">
              {error}
            </p>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading || !staffId.trim()}
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Verifying...
            </span>
          ) : (
            'Continue'
          )}
        </button>
      </form>

      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h3 className="text-sm font-semibold text-blue-900 mb-2">ID Format Guidelines:</h3>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• 3 letters followed by numbers</li>
          <li>• Examples: EMP12345, STF0001, MGR9876</li>
          <li>• Minimum 7 characters total</li>
        </ul>
      </div>
    </div>
  );
};

export default StaffIDInput;
