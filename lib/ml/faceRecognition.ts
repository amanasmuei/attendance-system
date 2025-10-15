/**
 * Face Recognition Module
 * Handles face descriptor extraction and matching
 */

import * as faceapi from '@vladmandic/face-api';
import { modelLoader } from './modelLoader';

export interface FaceDescriptor {
  descriptor: Float32Array;
  userId: string;
  timestamp: Date;
  confidence: number;
}

export interface MatchResult {
  userId: string;
  distance: number;
  confidence: number;
  isMatch: boolean;
}

export interface RecognitionOptions {
  matchThreshold?: number;
  minConfidence?: number;
}

const DEFAULT_OPTIONS: Required<RecognitionOptions> = {
  matchThreshold: 0.6,
  minConfidence: 0.9,
};

/**
 * Extract face descriptor (128-dimensional embedding) from image
 */
export async function extractFaceDescriptor(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
): Promise<Float32Array | null> {
  // Ensure models are loaded
  if (!modelLoader.isLoaded('recognition')) {
    throw new Error('Recognition models not loaded. Call modelLoader.loadModels("recognition") first.');
  }

  // Detect face with landmarks and compute descriptor
  const detection = await faceapi
    .detectSingleFace(input, new faceapi.SsdMobilenetv1Options())
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) {
    return null;
  }

  return detection.descriptor;
}

/**
 * Extract multiple face descriptors from multiple samples
 */
export async function extractMultipleDescriptors(
  inputs: (HTMLImageElement | HTMLVideoElement | HTMLCanvasElement)[],
): Promise<Float32Array[]> {
  const descriptors: Float32Array[] = [];

  for (const input of inputs) {
    const descriptor = await extractFaceDescriptor(input);
    if (descriptor) {
      descriptors.push(descriptor);
    }
  }

  return descriptors;
}

/**
 * Calculate Euclidean distance between two face descriptors
 */
export function calculateDistance(descriptor1: Float32Array, descriptor2: Float32Array): number {
  if (descriptor1.length !== descriptor2.length) {
    throw new Error('Descriptors must have the same length');
  }

  let sum = 0;
  for (let i = 0; i < descriptor1.length; i++) {
    const diff = descriptor1[i] - descriptor2[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

/**
 * Find best match from a list of enrolled face descriptors
 */
export function findBestMatch(
  queryDescriptor: Float32Array,
  enrolledDescriptors: FaceDescriptor[],
  options: RecognitionOptions = {},
): MatchResult | null {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  let bestMatch: MatchResult | null = null;
  let minDistance = Infinity;

  for (const enrolled of enrolledDescriptors) {
    const distance = calculateDistance(queryDescriptor, enrolled.descriptor);

    if (distance < minDistance) {
      minDistance = distance;
      bestMatch = {
        userId: enrolled.userId,
        distance,
        confidence: 1 - distance,
        isMatch: distance < opts.matchThreshold,
      };
    }
  }

  // Only return if it's a valid match
  if (bestMatch && bestMatch.isMatch) {
    return bestMatch;
  }

  return null;
}

/**
 * Find all matches above threshold
 */
export function findAllMatches(
  queryDescriptor: Float32Array,
  enrolledDescriptors: FaceDescriptor[],
  options: RecognitionOptions = {},
): MatchResult[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  const matches: MatchResult[] = [];

  for (const enrolled of enrolledDescriptors) {
    const distance = calculateDistance(queryDescriptor, enrolled.descriptor);

    if (distance < opts.matchThreshold) {
      matches.push({
        userId: enrolled.userId,
        distance,
        confidence: 1 - distance,
        isMatch: true,
      });
    }
  }

  // Sort by distance (lowest first)
  matches.sort((a, b) => a.distance - b.distance);

  return matches;
}

/**
 * Compute average descriptor from multiple samples
 */
export function computeAverageDescriptor(descriptors: Float32Array[]): Float32Array {
  if (descriptors.length === 0) {
    throw new Error('No descriptors provided');
  }

  const length = descriptors[0].length;
  const average = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (const descriptor of descriptors) {
      sum += descriptor[i];
    }
    average[i] = sum / descriptors.length;
  }

  return average;
}

/**
 * Validate descriptor quality
 */
export function validateDescriptor(descriptor: Float32Array): {
  isValid: boolean;
  messages: string[];
} {
  const messages: string[] = [];
  let isValid = true;

  // Check length
  if (descriptor.length !== 128) {
    isValid = false;
    messages.push(`Invalid descriptor length: ${descriptor.length} (expected 128)`);
  }

  // Check for NaN or Infinity
  for (let i = 0; i < descriptor.length; i++) {
    if (!isFinite(descriptor[i])) {
      isValid = false;
      messages.push(`Invalid value at index ${i}`);
      break;
    }
  }

  // Check if all zeros
  const allZeros = descriptor.every((val) => val === 0);
  if (allZeros) {
    isValid = false;
    messages.push('Descriptor contains all zeros');
  }

  return { isValid, messages };
}

/**
 * Convert descriptor to base64 string for storage
 */
export function descriptorToBase64(descriptor: Float32Array): string {
  const bytes = new Uint8Array(descriptor.buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Convert base64 string back to descriptor
 */
export function base64ToDescriptor(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

/**
 * Create a face matcher for continuous recognition
 */
export class FaceMatcher {
  private enrolledDescriptors: FaceDescriptor[] = [];
  private options: Required<RecognitionOptions>;

  constructor(enrolledDescriptors: FaceDescriptor[] = [], options: RecognitionOptions = {}) {
    this.enrolledDescriptors = enrolledDescriptors;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Add enrolled descriptor
   */
  public addDescriptor(descriptor: FaceDescriptor): void {
    this.enrolledDescriptors.push(descriptor);
  }

  /**
   * Remove descriptor by userId
   */
  public removeDescriptor(userId: string): void {
    this.enrolledDescriptors = this.enrolledDescriptors.filter((d) => d.userId !== userId);
  }

  /**
   * Match a face against enrolled descriptors
   */
  public async matchFace(
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  ): Promise<MatchResult | null> {
    const descriptor = await extractFaceDescriptor(input);
    if (!descriptor) {
      return null;
    }

    return findBestMatch(descriptor, this.enrolledDescriptors, this.options);
  }

  /**
   * Match descriptor directly
   */
  public matchDescriptor(descriptor: Float32Array): MatchResult | null {
    return findBestMatch(descriptor, this.enrolledDescriptors, this.options);
  }

  /**
   * Get all enrolled user IDs
   */
  public getEnrolledUserIds(): string[] {
    return Array.from(new Set(this.enrolledDescriptors.map((d) => d.userId)));
  }

  /**
   * Get descriptor count
   */
  public getDescriptorCount(): number {
    return this.enrolledDescriptors.length;
  }

  /**
   * Update match threshold
   */
  public setMatchThreshold(threshold: number): void {
    this.options.matchThreshold = threshold;
  }

  /**
   * Get current match threshold
   */
  public getMatchThreshold(): number {
    return this.options.matchThreshold;
  }
}
