'use client';

/**
 * Reusable Camera Capture Component
 * Provides webcam access with face detection overlay
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { detectSingleFace, drawDetection, FaceDetectionResult } from '@/lib/ml/faceDetection';
import { modelLoader } from '@/lib/ml/modelLoader';

export interface CameraCaptureProps {
  onCapture?: (imageSrc: string, detection: FaceDetectionResult | null) => void;
  onFaceDetected?: (detection: FaceDetectionResult) => void;
  showOverlay?: boolean;
  showValidation?: boolean;
  autoCapture?: boolean;
  autoCaptureDelay?: number;
  mirrored?: boolean;
  className?: string;
}

export const CameraCapture: React.FC<CameraCaptureProps> = ({
  onCapture,
  onFaceDetected,
  showOverlay = true,
  showValidation = true,
  autoCapture = false,
  autoCaptureDelay = 3000,
  mirrored = true,
  className = '',
}) => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);

  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [currentDetection, setCurrentDetection] = useState<FaceDetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  // Load models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        if (!modelLoader.isLoaded('detection')) {
          await modelLoader.loadModels('detection');
        }
        setIsModelLoaded(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load models');
        console.error('Model loading error:', err);
      }
    };

    loadModels();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Handle manual capture
  const handleCapture = useCallback(() => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc && onCapture) {
      onCapture(imageSrc, currentDetection);
    }
  }, [onCapture, currentDetection]);

  // Face detection loop
  const detectFace = useCallback(async () => {
    if (!webcamRef.current || !canvasRef.current || !isModelLoaded) {
      return;
    }

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;

    if (!video || video.readyState !== 4) {
      animationRef.current = requestAnimationFrame(detectFace);
      return;
    }

    try {
      // Detect face
      const detection = await detectSingleFace(video);

      // Update canvas size to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Clear canvas
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      if (detection && showOverlay && ctx) {
        // Draw detection overlay
        drawDetection(canvas, detection);

        // Show validation messages
        if (showValidation && !detection.isValid) {
          ctx.fillStyle = '#ef4444';
          ctx.font = '14px Arial';
          let yOffset = 30;
          for (const message of detection.validationMessages) {
            ctx.fillText(message, 10, yOffset);
            yOffset += 20;
          }
        }
      }

      setCurrentDetection(detection);

      // Notify parent component
      if (detection && onFaceDetected) {
        onFaceDetected(detection);
      }

      // Auto capture if enabled and face is valid
      if (autoCapture && detection && detection.isValid && !isCapturing) {
        setIsCapturing(true);
        setTimeout(() => {
          handleCapture();
          setIsCapturing(false);
        }, autoCaptureDelay);
      }
    } catch (err) {
      console.error('Detection error:', err);
    }

    // Continue detection loop
    animationRef.current = requestAnimationFrame(detectFace);
  }, [
    isModelLoaded,
    showOverlay,
    showValidation,
    onFaceDetected,
    autoCapture,
    autoCaptureDelay,
    isCapturing,
    handleCapture,
  ]);

  // Start detection when model is loaded
  useEffect(() => {
    if (isModelLoaded && webcamRef.current) {
      animationRef.current = requestAnimationFrame(detectFace);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isModelLoaded, detectFace]);

  // Handle camera errors
  const handleUserMediaError = (err: string | DOMException) => {
    console.error('Camera error:', err);
    if (typeof err === 'string') {
      setError(err);
    } else {
      setError(err.message);
    }
  };

  if (error) {
    return (
      <div className="flex items-center justify-center p-8 bg-red-50 border border-red-200 rounded-lg">
        <div className="text-center">
          <p className="text-red-600 font-semibold mb-2">Camera Error</p>
          <p className="text-sm text-red-500">{error}</p>
        </div>
      </div>
    );
  }

  if (!isModelLoaded) {
    return (
      <div className="flex items-center justify-center p-8 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading face detection models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{
            width: 1280,
            height: 720,
            facingMode: 'user',
          }}
          mirrored={mirrored}
          className="w-full h-auto rounded-lg"
          onUserMediaError={handleUserMediaError}
        />

        {showOverlay && (
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
        )}
      </div>

      {/* Status indicator */}
      <div className="absolute top-4 right-4 flex items-center gap-2 bg-black/50 text-white px-3 py-2 rounded-lg text-sm">
        <div
          className={`w-2 h-2 rounded-full ${
            currentDetection && currentDetection.isValid ? 'bg-green-500' : 'bg-red-500'
          }`}
        />
        {currentDetection ? (
          currentDetection.isValid ? (
            <span>Face detected</span>
          ) : (
            <span>Adjust position</span>
          )
        ) : (
          <span>Looking for face...</span>
        )}
      </div>

      {/* Capture button (if not auto-capture) */}
      {!autoCapture && (
        <div className="mt-4 text-center">
          <button
            onClick={handleCapture}
            disabled={!currentDetection || !currentDetection.isValid}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Capture Photo
          </button>
        </div>
      )}

      {/* Face quality feedback */}
      {showValidation && currentDetection && !currentDetection.isValid && (
        <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="font-semibold text-yellow-800 mb-2">Please adjust:</p>
          <ul className="list-disc list-inside text-sm text-yellow-700">
            {currentDetection.validationMessages.map((message, index) => (
              <li key={index}>{message}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default CameraCapture;
