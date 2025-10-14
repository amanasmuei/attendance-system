'use client';

export default function OfflinePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 p-4">
      <div className="max-w-md text-center">
        <div className="mb-6">
          <svg
            className="mx-auto h-24 w-24 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414"
            />
          </svg>
        </div>

        <h1 className="mb-3 text-3xl font-bold text-gray-900">You&apos;re Offline</h1>

        <p className="mb-6 text-gray-600">
          This app requires an internet connection to load new content. Don&apos;t worry - your
          attendance records are saved locally and will sync when you&apos;re back online.
        </p>

        <div className="rounded-lg bg-blue-50 p-4">
          <p className="text-sm text-blue-900">
            <strong>Offline Features Available:</strong>
            <br />
            • View enrolled users
            <br />
            • Check-in with face recognition
            <br />
            • Record attendance locally
            <br />• Access previously loaded events
          </p>
        </div>

        <button
          onClick={() => window.location.reload()}
          className="mt-6 rounded-md bg-black px-6 py-3 text-white transition-colors hover:bg-gray-800"
        >
          Try Again
        </button>
      </div>
    </div>
  );
}
