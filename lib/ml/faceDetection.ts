/**
 * Face Detection Module
 * Provides face detection functionality with quality validation
 */

import * as faceapi from '@vladmandic/face-api';
import { modelLoader } from './modelLoader';

export interface DetectionOptions {
  minConfidence?: number;
  minFaceSize?: number;
  inputSize?: number;
  scoreThreshold?: number;
}

export interface FaceDetectionResult {
  detection: faceapi.FaceDetection;
  landmarks?: faceapi.FaceLandmarks68;
  isValid: boolean;
  validationMessages: string[];
  confidence: number;
  faceSize: { width: number; height: number };
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

const DEFAULT_OPTIONS: Required<DetectionOptions> = {
  minConfidence: 0.9,
  minFaceSize: 80,
  inputSize: 512,
  scoreThreshold: 0.5,
};

/**
 * Detect a single face in an image with quality validation
 */
export async function detectSingleFace(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  options: DetectionOptions = {},
): Promise<FaceDetectionResult | null> {
  // Ensure models are loaded
  if (!modelLoader.isLoaded('detection')) {
    throw new Error('Detection model not loaded. Call modelLoader.loadModels() first.');
  }

  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Detect face with landmarks
  const detectionWithLandmarks = await faceapi
    .detectSingleFace(
      input,
      new faceapi.SsdMobilenetv1Options({
        minConfidence: opts.scoreThreshold,
      }),
    )
    .withFaceLandmarks();

  if (!detectionWithLandmarks) {
    return null;
  }

  const { detection, landmarks } = detectionWithLandmarks;
  const box = detection.box;
  const confidence = detection.score;

  // Validate face quality
  const validationMessages: string[] = [];
  let isValid = true;

  // Check confidence
  if (confidence < opts.minConfidence) {
    isValid = false;
    validationMessages.push(
      `Low confidence: ${(confidence * 100).toFixed(1)}% (minimum: ${opts.minConfidence * 100}%)`,
    );
  }

  // Check face size
  const faceSize = { width: box.width, height: box.height };
  if (box.width < opts.minFaceSize || box.height < opts.minFaceSize) {
    isValid = false;
    validationMessages.push(
      `Face too small: ${Math.min(box.width, box.height).toFixed(0)}px (minimum: ${opts.minFaceSize}px)`,
    );
  }

  // Check if face is too close to edges
  if (input instanceof HTMLVideoElement || input instanceof HTMLImageElement) {
    const margin = 20;
    if (
      box.x < margin ||
      box.y < margin ||
      box.x + box.width > input.width - margin ||
      box.y + box.height > input.height - margin
    ) {
      validationMessages.push('Face too close to edge - please center your face');
    }
  }

  return {
    detection,
    landmarks,
    isValid,
    validationMessages,
    confidence,
    faceSize,
  };
}

/**
 * Detect all faces in an image
 */
export async function detectAllFaces(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  options: DetectionOptions = {},
): Promise<FaceDetectionResult[]> {
  // Ensure models are loaded
  if (!modelLoader.isLoaded('detection')) {
    throw new Error('Detection model not loaded. Call modelLoader.loadModels() first.');
  }

  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Detect all faces with landmarks
  const detections = await faceapi
    .detectAllFaces(
      input,
      new faceapi.SsdMobilenetv1Options({
        minConfidence: opts.scoreThreshold,
      }),
    )
    .withFaceLandmarks();

  return detections.map((detectionWithLandmarks) => {
    const { detection, landmarks } = detectionWithLandmarks;
    const box = detection.box;
    const confidence = detection.score;

    const validationMessages: string[] = [];
    let isValid = true;

    if (confidence < opts.minConfidence) {
      isValid = false;
      validationMessages.push(`Low confidence: ${(confidence * 100).toFixed(1)}%`);
    }

    const faceSize = { width: box.width, height: box.height };
    if (box.width < opts.minFaceSize || box.height < opts.minFaceSize) {
      isValid = false;
      validationMessages.push(`Face too small: ${Math.min(box.width, box.height).toFixed(0)}px`);
    }

    return {
      detection,
      landmarks,
      isValid,
      validationMessages,
      confidence,
      faceSize,
    };
  });
}

/**
 * Draw face detection on canvas
 */
export function drawDetection(
  canvas: HTMLCanvasElement,
  detection: FaceDetectionResult,
  options: {
    drawBox?: boolean;
    drawLandmarks?: boolean;
    boxColor?: string;
    landmarkColor?: string;
  } = {},
): void {
  const { drawBox = true, drawLandmarks = true, boxColor = '#00ff00', landmarkColor = '#ff0000' } = options;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Draw bounding box
  if (drawBox) {
    const box = detection.detection.box;
    ctx.strokeStyle = detection.isValid ? boxColor : '#ff0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw confidence
    ctx.fillStyle = detection.isValid ? boxColor : '#ff0000';
    ctx.font = '16px Arial';
    ctx.fillText(`${(detection.confidence * 100).toFixed(1)}%`, box.x, box.y - 5);
  }

  // Draw landmarks
  if (drawLandmarks && detection.landmarks) {
    const landmarks = detection.landmarks.positions;
    ctx.fillStyle = landmarkColor;
    landmarks.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
}

/**
 * Get face bounding box from detection
 */
export function getBoundingBox(detection: FaceDetectionResult): BoundingBox {
  const box = detection.detection.box;
  return {
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
  };
}

/**
 * Check if face is centered in frame
 */
export function isFaceCentered(
  detection: FaceDetectionResult,
  frameWidth: number,
  frameHeight: number,
  tolerance: number = 0.2,
): boolean {
  const box = detection.detection.box;
  const faceCenterX = box.x + box.width / 2;
  const faceCenterY = box.y + box.height / 2;
  const frameCenterX = frameWidth / 2;
  const frameCenterY = frameHeight / 2;

  const maxOffsetX = frameWidth * tolerance;
  const maxOffsetY = frameHeight * tolerance;

  const offsetX = Math.abs(faceCenterX - frameCenterX);
  const offsetY = Math.abs(faceCenterY - frameCenterY);

  return offsetX <= maxOffsetX && offsetY <= maxOffsetY;
}

/**
 * Extract face region from image
 */
export function extractFaceRegion(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  detection: FaceDetectionResult,
  padding: number = 0.2,
): HTMLCanvasElement {
  const box = detection.detection.box;

  // Calculate padded box
  const paddingX = box.width * padding;
  const paddingY = box.height * padding;
  const x = Math.max(0, box.x - paddingX);
  const y = Math.max(0, box.y - paddingY);
  const width = Math.min(
    input instanceof HTMLVideoElement ? input.videoWidth : input.width,
    box.width + 2 * paddingX,
  );
  const height = Math.min(
    input instanceof HTMLVideoElement ? input.videoHeight : input.height,
    box.height + 2 * paddingY,
  );

  // Create canvas with face region
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    ctx.drawImage(input, x, y, width, height, 0, 0, width, height);
  }

  return canvas;
}
