/**
 * Face Matching Service for Recognition
 * Handles face descriptor comparison and user identification
 */

import * as faceapi from 'face-api.js';

export interface FaceDescriptorData {
  userId: string;
  userName: string;
  descriptors: Float32Array[];
  enrolledAt: number;
}

export interface MatchResult {
  userId: string | null;
  userName: string | null;
  distance: number;
  confidence: number;
  isMatch: boolean;
}

export class FaceMatcher {
  private matcher: faceapi.FaceMatcher | null = null;
  private knownFaces: Map<string, FaceDescriptorData> = new Map();
  private distanceThreshold: number;

  constructor(distanceThreshold = 0.6) {
    this.distanceThreshold = distanceThreshold;
  }

  /**
   * Add a known face to the matcher
   */
  async addFace(data: FaceDescriptorData): Promise<void> {
    // Store in our map
    this.knownFaces.set(data.userId, data);

    // Rebuild matcher
    await this.rebuildMatcher();
  }

  /**
   * Add multiple known faces
   */
  async addFaces(faces: FaceDescriptorData[]): Promise<void> {
    faces.forEach((face) => {
      this.knownFaces.set(face.userId, face);
    });

    await this.rebuildMatcher();
  }

  /**
   * Remove a face from the matcher
   */
  async removeFace(userId: string): Promise<boolean> {
    const deleted = this.knownFaces.delete(userId);

    if (deleted) {
      await this.rebuildMatcher();
    }

    return deleted;
  }

  /**
   * Clear all known faces
   */
  clearFaces(): void {
    this.knownFaces.clear();
    this.matcher = null;
  }

  /**
   * Get all known faces
   */
  getKnownFaces(): FaceDescriptorData[] {
    return Array.from(this.knownFaces.values());
  }

  /**
   * Get face by user ID
   */
  getFaceByUserId(userId: string): FaceDescriptorData | undefined {
    return this.knownFaces.get(userId);
  }

  /**
   * Match a face descriptor against known faces
   */
  matchFace(descriptor: Float32Array): MatchResult {
    if (!this.matcher) {
      return {
        userId: null,
        userName: null,
        distance: 1.0,
        confidence: 0,
        isMatch: false,
      };
    }

    try {
      const bestMatch = this.matcher.findBestMatch(descriptor);

      // Check if it's an unknown face
      if (bestMatch.label === 'unknown') {
        return {
          userId: null,
          userName: null,
          distance: bestMatch.distance,
          confidence: 0,
          isMatch: false,
        };
      }

      // Get user data
      const userData = this.knownFaces.get(bestMatch.label);

      // Calculate confidence (inverse of distance, normalized)
      const confidence = Math.max(0, 1 - bestMatch.distance);
      const isMatch = bestMatch.distance < this.distanceThreshold;

      return {
        userId: bestMatch.label,
        userName: userData?.userName || bestMatch.label,
        distance: bestMatch.distance,
        confidence,
        isMatch,
      };
    } catch (error) {
      console.error('Face matching error:', error);
      return {
        userId: null,
        userName: null,
        distance: 1.0,
        confidence: 0,
        isMatch: false,
      };
    }
  }

  /**
   * Find all potential matches (not just best match)
   */
  findAllMatches(descriptor: Float32Array, maxDistance?: number): MatchResult[] {
    if (!this.matcher) {
      return [];
    }

    const threshold = maxDistance || this.distanceThreshold;
    const results: MatchResult[] = [];

    // Compare against all known faces
    for (const [userId, faceData] of this.knownFaces.entries()) {
      for (const knownDescriptor of faceData.descriptors) {
        const distance = faceapi.euclideanDistance(descriptor, knownDescriptor);

        if (distance < threshold) {
          const confidence = Math.max(0, 1 - distance);

          results.push({
            userId,
            userName: faceData.userName,
            distance,
            confidence,
            isMatch: true,
          });
        }
      }
    }

    // Sort by distance (best matches first)
    return results.sort((a, b) => a.distance - b.distance);
  }

  /**
   * Update distance threshold
   */
  setDistanceThreshold(threshold: number): void {
    this.distanceThreshold = threshold;
    // Note: FaceMatcher doesn't support updating threshold dynamically
    // Results are still valid, just isMatch interpretation changes
  }

  /**
   * Get current distance threshold
   */
  getDistanceThreshold(): number {
    return this.distanceThreshold;
  }

  /**
   * Check if matcher is ready
   */
  isReady(): boolean {
    return this.matcher !== null && this.knownFaces.size > 0;
  }

  /**
   * Get number of known faces
   */
  getKnownFacesCount(): number {
    return this.knownFaces.size;
  }

  /**
   * Calculate similarity percentage between two descriptors
   */
  calculateSimilarity(descriptor1: Float32Array, descriptor2: Float32Array): number {
    const distance = faceapi.euclideanDistance(descriptor1, descriptor2);
    return Math.max(0, (1 - distance) * 100);
  }

  /**
   * Rebuild the FaceMatcher with current known faces
   */
  private async rebuildMatcher(): Promise<void> {
    if (this.knownFaces.size === 0) {
      this.matcher = null;
      return;
    }

    try {
      // Create labeled descriptors for face-api.js
      const labeledDescriptors: faceapi.LabeledFaceDescriptors[] = [];

      for (const [userId, faceData] of this.knownFaces.entries()) {
        labeledDescriptors.push(
          new faceapi.LabeledFaceDescriptors(userId, faceData.descriptors)
        );
      }

      // Create new matcher
      this.matcher = new faceapi.FaceMatcher(labeledDescriptors, this.distanceThreshold);
    } catch (error) {
      console.error('Error rebuilding face matcher:', error);
      throw new Error(`Failed to rebuild face matcher: ${error}`);
    }
  }
}

// Singleton instance
let faceMatcherInstance: FaceMatcher | null = null;

export function getFaceMatcher(distanceThreshold?: number): FaceMatcher {
  if (!faceMatcherInstance) {
    faceMatcherInstance = new FaceMatcher(distanceThreshold);
  }
  return faceMatcherInstance;
}

/**
 * Utility function to compare two face descriptors
 */
export function compareFaceDescriptors(
  descriptor1: Float32Array,
  descriptor2: Float32Array
): { distance: number; similarity: number } {
  const distance = faceapi.euclideanDistance(descriptor1, descriptor2);
  const similarity = Math.max(0, (1 - distance) * 100);

  return { distance, similarity };
}

/**
 * Check if two descriptors match within threshold
 */
export function descriptorsMatch(
  descriptor1: Float32Array,
  descriptor2: Float32Array,
  threshold = 0.6
): boolean {
  const distance = faceapi.euclideanDistance(descriptor1, descriptor2);
  return distance < threshold;
}
