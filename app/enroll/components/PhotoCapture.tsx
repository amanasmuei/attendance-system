'use client';

/**
 * Multi-Sample Photo Capture Component
 * Captures multiple face photos for enrollment
 */

import React, { useState } from 'react';
import Image from 'next/image';
import CameraCapture from '@/components/CameraCapture';
import { FaceDetectionResult } from '@/lib/ml/faceDetection';

export interface PhotoCaptureProps {
  requiredSamples?: number;
  onComplete: (samples: string[]) => void;
  userName?: string;
}

export const PhotoCapture: React.FC<PhotoCaptureProps> = ({
  requiredSamples = 3,
  onComplete,
  userName,
}) => {
  const [samples, setSamples] = useState<string[]>([]);
  const [currentDetection, setCurrentDetection] = useState<FaceDetectionResult | null>(null);

  const handleCapture = (imageSrc: string, detection: FaceDetectionResult | null) => {
    if (detection && detection.isValid) {
      const newSamples = [...samples, imageSrc];
      setSamples(newSamples);

      if (newSamples.length >= requiredSamples) {
        onComplete(newSamples);
      }
    }
  };

  const handleFaceDetected = (detection: FaceDetectionResult) => {
    setCurrentDetection(detection);
  };

  const handleRemoveSample = (index: number) => {
    setSamples(samples.filter((_, i) => i !== index));
  };

  const progress = (samples.length / requiredSamples) * 100;

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6 text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Face Enrollment {userName && `for ${userName}`}
        </h2>
        <p className="text-gray-600">
          Capture {requiredSamples} photos from different angles for better recognition
        </p>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">
            Progress: {samples.length} of {requiredSamples} photos
          </span>
          <span className="text-sm font-medium text-gray-700">{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {samples.length < requiredSamples ? (
        <>
          {/* Instructions */}
          <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="text-sm font-semibold text-blue-900 mb-2">
              Capture Photo {samples.length + 1}:
            </h3>
            <ul className="text-sm text-blue-700 space-y-1">
              {samples.length === 0 && <li>• Look directly at the camera</li>}
              {samples.length === 1 && <li>• Turn your head slightly to the left</li>}
              {samples.length === 2 && <li>• Turn your head slightly to the right</li>}
              <li>• Ensure good lighting</li>
              <li>• Keep a neutral expression</li>
              <li>• Remove glasses if possible</li>
            </ul>
          </div>

          {/* Camera */}
          <CameraCapture
            onCapture={handleCapture}
            onFaceDetected={handleFaceDetected}
            showOverlay={true}
            showValidation={true}
          />

          {/* Quality Feedback */}
          {currentDetection && currentDetection.isValid && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-green-800 font-semibold">
                ✓ Face detected with high quality ({(currentDetection.confidence * 100).toFixed(1)}% confidence)
              </p>
              <p className="text-sm text-green-700 mt-1">Click &ldquo;Capture Photo&rdquo; when ready</p>
            </div>
          )}
        </>
      ) : (
        <div className="text-center p-8 bg-green-50 border border-green-200 rounded-lg">
          <div className="text-green-600 text-5xl mb-4">✓</div>
          <h3 className="text-xl font-bold text-green-900 mb-2">All Photos Captured!</h3>
          <p className="text-green-700">Processing your enrollment...</p>
        </div>
      )}

      {/* Captured Samples Preview */}
      {samples.length > 0 && samples.length < requiredSamples && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Captured Photos:</h3>
          <div className="grid grid-cols-3 gap-4">
            {samples.map((sample, index) => (
              <div key={index} className="relative group">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={sample}
                  alt={`Sample ${index + 1}`}
                  className="w-full h-auto rounded-lg border-2 border-green-500"
                />
                <div className="absolute top-2 left-2 bg-green-500 text-white text-xs font-bold px-2 py-1 rounded">
                  Photo {index + 1}
                </div>
                <button
                  onClick={() => handleRemoveSample(index)}
                  className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Remove photo"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PhotoCapture;
