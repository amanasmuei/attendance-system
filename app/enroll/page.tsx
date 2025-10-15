'use client';

/**
 * Face Enrollment Page
 * Multi-step enrollment flow with Staff ID validation and face capture
 */

import React, { useState } from 'react';
import Link from 'next/link';
import StaffIDInput from './components/StaffIDInput';
import PhotoCapture from './components/PhotoCapture';
import { extractMultipleDescriptors, computeAverageDescriptor } from '@/lib/ml/faceRecognition';
import { generateKey, encryptDescriptor, exportKey } from '@/lib/crypto/encryption';
import { modelLoader } from '@/lib/ml/modelLoader';

enum EnrollmentStep {
  STAFF_ID = 'staff_id',
  PHOTO_CAPTURE = 'photo_capture',
  PROCESSING = 'processing',
  SUCCESS = 'success',
  ERROR = 'error',
}

interface UserData {
  staffId: string;
  name?: string;
  email?: string;
}

export default function EnrollPage() {
  const [step, setStep] = useState<EnrollmentStep>(EnrollmentStep.STAFF_ID);
  const [userData, setUserData] = useState<UserData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);

  /**
   * Step 1: Verify Staff ID
   */
  const handleStaffIdSubmit = async (staffId: string) => {
    setIsVerifying(true);
    setError(null);

    try {
      // TODO: Call API to verify staff ID
      // For now, simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Mock user data - replace with actual API response
      const mockUserData: UserData = {
        staffId,
        name: `User ${staffId}`,
        email: `${staffId.toLowerCase()}@company.com`,
      };

      setUserData(mockUserData);

      // Load ML models before photo capture
      if (!modelLoader.isLoaded('recognition')) {
        await modelLoader.loadModels('recognition');
      }

      setStep(EnrollmentStep.PHOTO_CAPTURE);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to verify Staff ID');
      console.error('Staff ID verification error:', err);
    } finally {
      setIsVerifying(false);
    }
  };

  /**
   * Step 2: Process captured photos
   */
  const handlePhotosComplete = async (samples: string[]) => {
    setStep(EnrollmentStep.PROCESSING);
    setError(null);

    try {
      // Convert base64 images to HTMLImageElement
      const images = await Promise.all(
        samples.map(
          (sample) =>
            new Promise<HTMLImageElement>((resolve, reject) => {
              const img = new Image();
              img.onload = () => resolve(img);
              img.onerror = reject;
              img.src = sample;
            }),
        ),
      );

      // Extract face descriptors from all samples
      const descriptors = await extractMultipleDescriptors(images);

      if (descriptors.length === 0) {
        throw new Error('No valid face descriptors extracted');
      }

      if (descriptors.length < samples.length) {
        console.warn(
          `Only ${descriptors.length} of ${samples.length} samples produced valid descriptors`,
        );
      }

      // Compute average descriptor
      const averageDescriptor = computeAverageDescriptor(descriptors);

      // Generate encryption key
      const encryptionKey = await generateKey();

      // Encrypt descriptor
      const { encrypted, iv } = await encryptDescriptor(averageDescriptor, encryptionKey);

      // Export key for storage
      const exportedKey = await exportKey(encryptionKey);

      // Prepare enrollment data
      const enrollmentData = {
        userId: userData!.staffId,
        encryptedDescriptor: encrypted,
        encryptionIV: iv,
        encryptionKey: exportedKey,
        sampleCount: descriptors.length,
        enrolledAt: new Date().toISOString(),
      };

      // TODO: Send to API endpoint
      console.log('Enrollment data:', enrollmentData);

      // Save to IndexedDB for offline use
      // TODO: Implement IndexedDB storage

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1500));

      setStep(EnrollmentStep.SUCCESS);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process enrollment');
      setStep(EnrollmentStep.ERROR);
      console.error('Enrollment processing error:', err);
    }
  };

  /**
   * Restart enrollment
   */
  const handleRestart = () => {
    setStep(EnrollmentStep.STAFF_ID);
    setUserData(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Face Enrollment</h1>
          <p className="text-gray-600">Enroll your face for contactless attendance</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center">
            {/* Step 1: Staff ID */}
            <div className="flex items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step === EnrollmentStep.STAFF_ID
                    ? 'bg-blue-600 text-white'
                    : step === EnrollmentStep.PHOTO_CAPTURE ||
                        step === EnrollmentStep.PROCESSING ||
                        step === EnrollmentStep.SUCCESS
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-300 text-gray-600'
                }`}
              >
                1
              </div>
              <span className="ml-2 text-sm font-medium text-gray-700">Staff ID</span>
            </div>

            {/* Connector */}
            <div className="w-16 h-1 mx-2 bg-gray-300">
              <div
                className={`h-full ${
                  step !== EnrollmentStep.STAFF_ID ? 'bg-green-600' : 'bg-gray-300'
                }`}
              ></div>
            </div>

            {/* Step 2: Photo Capture */}
            <div className="flex items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step === EnrollmentStep.PHOTO_CAPTURE
                    ? 'bg-blue-600 text-white'
                    : step === EnrollmentStep.PROCESSING || step === EnrollmentStep.SUCCESS
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-300 text-gray-600'
                }`}
              >
                2
              </div>
              <span className="ml-2 text-sm font-medium text-gray-700">Capture Photos</span>
            </div>

            {/* Connector */}
            <div className="w-16 h-1 mx-2 bg-gray-300">
              <div
                className={`h-full ${step === EnrollmentStep.SUCCESS ? 'bg-green-600' : 'bg-gray-300'}`}
              ></div>
            </div>

            {/* Step 3: Complete */}
            <div className="flex items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step === EnrollmentStep.SUCCESS
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-300 text-gray-600'
                }`}
              >
                3
              </div>
              <span className="ml-2 text-sm font-medium text-gray-700">Complete</span>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          {step === EnrollmentStep.STAFF_ID && (
            <StaffIDInput onSubmit={handleStaffIdSubmit} isLoading={isVerifying} />
          )}

          {step === EnrollmentStep.PHOTO_CAPTURE && userData && (
            <PhotoCapture
              onComplete={handlePhotosComplete}
              userName={userData.name}
              requiredSamples={3}
            />
          )}

          {step === EnrollmentStep.PROCESSING && (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Processing Your Enrollment...
              </h3>
              <p className="text-gray-600">
                Extracting face features and encrypting your biometric data
              </p>
            </div>
          )}

          {step === EnrollmentStep.SUCCESS && (
            <div className="text-center py-12">
              <div className="text-green-600 text-6xl mb-4">✓</div>
              <h3 className="text-2xl font-bold text-green-900 mb-2">Enrollment Successful!</h3>
              <p className="text-gray-600 mb-6">
                You&apos;re all set! Your face has been enrolled for contactless attendance.
              </p>
              <div className="space-x-4">
                <Link
                  href="/"
                  className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Go to Home
                </Link>
                <button
                  onClick={handleRestart}
                  className="inline-block px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
                >
                  Enroll Another
                </button>
              </div>
            </div>
          )}

          {step === EnrollmentStep.ERROR && (
            <div className="text-center py-12">
              <div className="text-red-600 text-6xl mb-4">✗</div>
              <h3 className="text-2xl font-bold text-red-900 mb-2">Enrollment Failed</h3>
              <p className="text-gray-600 mb-6">{error}</p>
              <button
                onClick={handleRestart}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          )}
        </div>

        {/* Help Section */}
        <div className="mt-8 p-6 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">Privacy & Security</h3>
          <ul className="text-sm text-blue-700 space-y-2">
            <li>
              ✓ Your biometric data is encrypted using AES-256-GCM before storage
            </li>
            <li>✓ Face images are never stored - only mathematical representations</li>
            <li>✓ All processing happens locally in your browser for maximum privacy</li>
            <li>✓ You can withdraw consent and delete your data at any time</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
