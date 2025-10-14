/**
 * Face Detection Service using face-api.js
 * Handles model loading, face detection, landmarks, and emotion recognition
 */

import * as faceapi from 'face-api.js';

export interface FaceDetectionOptions {
  inputSize?: number;
  scoreThreshold?: number;
}

export interface Face Detection {
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  score: number;
  landmarks?: faceapi.FaceLandmarks68;
  descriptor?: Float32Array;
  expressions?: {
    neutral: number;
    happy: number;
    sad: number;
    angry: number;
    fearful: number;
    disgusted: number;
    surprised: number;
  };
  age?: number;
  gender?: string;
  genderProbability?: number;
}

export interface ModelLoadProgress {
  model: string;
  progress: number;
  status: 'loading' | 'loaded' | 'error';
}

export type ModelProgressCallback = (progress: ModelLoadProgress) => void;

/**
 * Face Detector class handles all face detection operations
 */
export class FaceDetector {
  private modelsLoaded = false;
  private modelsPath: string;
  private loadedModels: Set<string> = new Set();

  constructor(modelsPath: string = '/models') {
    this.modelsPath = modelsPath;
  }

  /**
   * Load all required models
   */
  async loadModels(
    includeRecognition = true,
    includeEmotions = true,
    onProgress?: ModelProgressCallback
  ): Promise<void> {
    if (this.modelsLoaded) {
      console.log('Models already loaded');
      return;
    }

    try {
      // Load TinyFaceDetector (required)
      onProgress?.({
        model: 'TinyFaceDetector',
        progress: 0,
        status: 'loading',
      });

      await faceapi.nets.tinyFaceDetector.loadFromUri(this.modelsPath);
      this.loadedModels.add('tinyFaceDetector');

      onProgress?.({
        model: 'TinyFaceDetector',
        progress: 100,
        status: 'loaded',
      });

      // Load Face Landmarks (required for recognition)
      onProgress?.({
        model: 'FaceLandmarks',
        progress: 0,
        status: 'loading',
      });

      await faceapi.nets.faceLandmark68Net.loadFromUri(this.modelsPath);
      this.loadedModels.add('faceLandmark68');

      onProgress?.({
        model: 'FaceLandmarks',
        progress: 100,
        status: 'loaded',
      });

      // Load Face Recognition model (optional)
      if (includeRecognition) {
        onProgress?.({
          model: 'FaceRecognition',
          progress: 0,
          status: 'loading',
        });

        await faceapi.nets.faceRecognitionNet.loadFromUri(this.modelsPath);
        this.loadedModels.add('faceRecognition');

        onProgress?.({
          model: 'FaceRecognition',
          progress: 100,
          status: 'loaded',
        });
      }

      // Load Face Expression model (optional)
      if (includeEmotions) {
        onProgress?.({
          model: 'FaceExpression',
          progress: 0,
          status: 'loading',
        });

        await faceapi.nets.faceExpressionNet.loadFromUri(this.modelsPath);
        this.loadedModels.add('faceExpression');

        onProgress?.({
          model: 'FaceExpression',
          progress: 100,
          status: 'loaded',
        });
      }

      this.modelsLoaded = true;
      console.log('All face-api.js models loaded successfully');
    } catch (error) {
      console.error('Error loading face-api.js models:', error);
      throw new Error(`Failed to load models: ${error}`);
    }
  }

  /**
   * Check if models are loaded
   */
  areModelsLoaded(): boolean {
    return this.modelsLoaded;
  }

  /**
   * Get list of loaded models
   */
  getLoadedModels(): string[] {
    return Array.from(this.loadedModels);
  }

  /**
   * Detect all faces in image or video element
   */
  async detectFaces(
    input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    options: FaceDetectionOptions = {}
  ): Promise<FaceDetection[]> {
    if (!this.modelsLoaded) {
      throw new Error('Models not loaded. Call loadModels() first.');
    }

    try {
      const detectionOptions = new faceapi.TinyFaceDetectorOptions({
        inputSize: options.inputSize || 416,
        scoreThreshold: options.scoreThreshold || 0.5,
      });

      // Build detection chain based on loaded models
      let detections = faceapi.detectAllFaces(input, detectionOptions);

      // Add landmarks if available
      if (this.loadedModels.has('faceLandmark68')) {
        detections = detections.withFaceLandmarks();
      }

      // Add descriptors if available
      if (this.loadedModels.has('faceRecognition')) {
        detections = detections.withFaceDescriptors();
      }

      // Add expressions if available
      if (this.loadedModels.has('faceExpression')) {
        detections = detections.withFaceExpressions();
      }

      const results = await detections;

      // Convert to our FaceDetection format
      return results.map((result) => ({
        box: {
          x: result.detection.box.x,
          y: result.detection.box.y,
          width: result.detection.box.width,
          height: result.detection.box.height,
        },
        score: result.detection.score,
        landmarks: result.landmarks,
        descriptor: result.descriptor,
        expressions: result.expressions
          ? {
              neutral: result.expressions.neutral,
              happy: result.expressions.happy,
              sad: result.expressions.sad,
              angry: result.expressions.angry,
              fearful: result.expressions.fearful,
              disgusted: result.expressions.disgusted,
              surprised: result.expressions.surprised,
            }
          : undefined,
      }));
    } catch (error) {
      console.error('Face detection error:', error);
      throw error;
    }
  }

  /**
   * Detect single face (returns first detected face)
   */
  async detectSingleFace(
    input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    options: FaceDetectionOptions = {}
  ): Promise<FaceDetection | null> {
    const faces = await this.detectFaces(input, options);
    return faces.length > 0 ? faces[0] : null;
  }

  /**
   * Get face descriptor for recognition
   */
  async getFaceDescriptor(
    input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
  ): Promise<Float32Array | null> {
    if (!this.loadedModels.has('faceRecognition')) {
      throw new Error('Face recognition model not loaded');
    }

    const face = await this.detectSingleFace(input);
    return face?.descriptor || null;
  }

  /**
   * Get dominant emotion from face
   */
  getDominantEmotion(expressions: FaceDetection['expressions']): {
    emotion: string;
    confidence: number;
  } | null {
    if (!expressions) return null;

    const emotions = Object.entries(expressions);
    const dominant = emotions.reduce((prev, current) =>
      current[1] > prev[1] ? current : prev
    );

    return {
      emotion: dominant[0],
      confidence: dominant[1],
    };
  }

  /**
   * Draw detection results on canvas
   */
  drawDetections(
    canvas: HTMLCanvasElement,
    detections: FaceDetection[],
    displaySize: { width: number; height: number }
  ): void {
    // Clear canvas
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Convert to face-api format for drawing
    const faceapiDetections = detections.map((det) => {
      const box = new faceapi.Box(det.box);
      const detection = new faceapi.FaceDetection(det.score, box, {});

      return {
        detection,
        landmarks: det.landmarks,
        expressions: det.expressions,
      };
    });

    // Resize results to match display
    const resized = faceapi.resizeResults(faceapiDetections, displaySize);

    // Draw boxes
    faceapi.draw.drawDetections(canvas, resized);

    // Draw landmarks if available
    if (detections[0]?.landmarks) {
      faceapi.draw.drawFaceLandmarks(canvas, resized);
    }

    // Draw expressions if available
    if (detections[0]?.expressions) {
      faceapi.draw.drawFaceExpressions(canvas, resized, 0.05);
    }
  }
}

// Singleton instance
let faceDetectorInstance: FaceDetector | null = null;

export function getFaceDetector(modelsPath?: string): FaceDetector {
  if (!faceDetectorInstance) {
    faceDetectorInstance = new FaceDetector(modelsPath);
  }
  return faceDetectorInstance;
}

/**
 * Initialize face detector with models
 */
export async function initializeFaceDetector(
  modelsPath = '/models',
  options: {
    includeRecognition?: boolean;
    includeEmotions?: boolean;
    onProgress?: ModelProgressCallback;
  } = {}
): Promise<FaceDetector> {
  const detector = getFaceDetector(modelsPath);

  if (!detector.areModelsLoaded()) {
    await detector.loadModels(
      options.includeRecognition !== false,
      options.includeEmotions !== false,
      options.onProgress
    );
  }

  return detector;
}
