/**
 * Emotion Detection Module
 * Detects and analyzes facial expressions (7 emotions)
 */

import * as faceapi from '@vladmandic/face-api';
import { modelLoader } from './modelLoader';

export type EmotionType = 'happy' | 'sad' | 'angry' | 'surprised' | 'fearful' | 'disgusted' | 'neutral';

export interface EmotionResult {
  emotion: EmotionType;
  confidence: number;
  allEmotions: Record<EmotionType, number>;
  timestamp: Date;
}

export interface EmotionDetectionOptions {
  minConfidence?: number;
}

const DEFAULT_OPTIONS: Required<EmotionDetectionOptions> = {
  minConfidence: 0.3,
};

/**
 * Detect emotion from a face in an image
 */
export async function detectEmotion(
  input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  options: EmotionDetectionOptions = {},
): Promise<EmotionResult | null> {
  // Ensure models are loaded
  if (!modelLoader.isLoaded('expression')) {
    throw new Error('Expression model not loaded. Call modelLoader.loadModels("expression") first.');
  }

  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Detect face with expression
  const detection = await faceapi
    .detectSingleFace(input, new faceapi.SsdMobilenetv1Options())
    .withFaceLandmarks()
    .withFaceExpressions();

  if (!detection || !detection.expressions) {
    return null;
  }

  const expressions = detection.expressions;

  // Map face-api expressions to our emotion types
  const allEmotions: Record<EmotionType, number> = {
    happy: expressions.happy,
    sad: expressions.sad,
    angry: expressions.angry,
    surprised: expressions.surprised,
    fearful: expressions.fearful,
    disgusted: expressions.disgusted,
    neutral: expressions.neutral,
  };

  // Find dominant emotion
  let dominantEmotion: EmotionType = 'neutral';
  let maxConfidence = 0;

  for (const [emotion, confidence] of Object.entries(allEmotions)) {
    if (confidence > maxConfidence) {
      maxConfidence = confidence;
      dominantEmotion = emotion as EmotionType;
    }
  }

  // Only return if confidence is above threshold
  if (maxConfidence < opts.minConfidence) {
    return null;
  }

  return {
    emotion: dominantEmotion,
    confidence: maxConfidence,
    allEmotions,
    timestamp: new Date(),
  };
}

/**
 * Get emotion message based on detected emotion
 */
export function getEmotionMessage(emotion: EmotionType): string {
  const messages: Record<EmotionType, string[]> = {
    happy: [
      'Welcome back! 😊 Great to see you smiling!',
      'Your smile brightens our day! 😊',
      'Looking happy today! Have a wonderful time! 🎉',
      'Welcome! Keep that beautiful smile! 😄',
    ],
    neutral: [
      'Welcome! Get ready to have fun! 🎉',
      'Great to see you! Enjoy the event! 👋',
      'Welcome aboard! 🚀',
      'Hello! Ready for an amazing experience? ✨',
    ],
    sad: [
      'Welcome! We hope this event brightens your day! 🌟',
      'Great to see you! Hope you find joy here today! 💫',
      "Welcome! We're here to make your day better! 🌈",
    ],
    surprised: [
      'Welcome! Hope we have some great surprises for you! 🎁',
      'Great to see you! Get ready for an exciting time! ⚡',
      'Welcome! Adventure awaits! 🚀',
    ],
    angry: [
      'Welcome! We hope to turn that frown upside down! 🌟',
      'Great to see you! Take a deep breath and enjoy! 🌸',
      "Welcome! Let's make this a positive experience! ✨",
    ],
    fearful: [
      "Welcome! You're in good hands here! 🤝",
      "Great to see you! Don't worry, you'll have a great time! 😊",
      'Welcome! Relax and enjoy the experience! 🌺',
    ],
    disgusted: [
      'Welcome! We promise a pleasant experience! 🌟',
      'Great to see you! Hope we exceed your expectations! ✨',
      "Welcome! Let's make this worthwhile! 🎯",
    ],
  };

  const emotionMessages = messages[emotion];
  const randomIndex = Math.floor(Math.random() * emotionMessages.length);
  return emotionMessages[randomIndex];
}

/**
 * Get emoji for emotion
 */
export function getEmotionEmoji(emotion: EmotionType): string {
  const emojis: Record<EmotionType, string> = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    surprised: '😲',
    fearful: '😨',
    disgusted: '🤢',
    neutral: '😐',
  };

  return emojis[emotion];
}

/**
 * Get emotion color (for UI visualization)
 */
export function getEmotionColor(emotion: EmotionType): string {
  const colors: Record<EmotionType, string> = {
    happy: '#10b981', // green
    sad: '#3b82f6', // blue
    angry: '#ef4444', // red
    surprised: '#f59e0b', // amber
    fearful: '#8b5cf6', // purple
    disgusted: '#84cc16', // lime
    neutral: '#6b7280', // gray
  };

  return colors[emotion];
}

/**
 * Analyze emotion trends over time
 */
export class EmotionAnalyzer {
  private history: EmotionResult[] = [];
  private maxHistorySize: number;

  constructor(maxHistorySize: number = 10) {
    this.maxHistorySize = maxHistorySize;
  }

  /**
   * Add emotion result to history
   */
  public addResult(result: EmotionResult): void {
    this.history.push(result);

    // Keep only recent results
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
    }
  }

  /**
   * Get dominant emotion over recent history
   */
  public getDominantEmotion(): EmotionType | null {
    if (this.history.length === 0) {
      return null;
    }

    const emotionCounts: Record<string, number> = {};

    for (const result of this.history) {
      emotionCounts[result.emotion] = (emotionCounts[result.emotion] || 0) + 1;
    }

    let dominantEmotion: EmotionType = 'neutral';
    let maxCount = 0;

    for (const [emotion, count] of Object.entries(emotionCounts)) {
      if (count > maxCount) {
        maxCount = count;
        dominantEmotion = emotion as EmotionType;
      }
    }

    return dominantEmotion;
  }

  /**
   * Get average confidence for each emotion
   */
  public getAverageEmotions(): Record<EmotionType, number> {
    if (this.history.length === 0) {
      return {
        happy: 0,
        sad: 0,
        angry: 0,
        surprised: 0,
        fearful: 0,
        disgusted: 0,
        neutral: 0,
      };
    }

    const sums: Record<EmotionType, number> = {
      happy: 0,
      sad: 0,
      angry: 0,
      surprised: 0,
      fearful: 0,
      disgusted: 0,
      neutral: 0,
    };

    for (const result of this.history) {
      for (const [emotion, confidence] of Object.entries(result.allEmotions)) {
        sums[emotion as EmotionType] += confidence;
      }
    }

    const averages: Record<EmotionType, number> = {} as Record<EmotionType, number>;
    for (const [emotion, sum] of Object.entries(sums)) {
      averages[emotion as EmotionType] = sum / this.history.length;
    }

    return averages;
  }

  /**
   * Clear history
   */
  public clearHistory(): void {
    this.history = [];
  }

  /**
   * Get history size
   */
  public getHistorySize(): number {
    return this.history.length;
  }

  /**
   * Get most recent emotion
   */
  public getLatestEmotion(): EmotionResult | null {
    if (this.history.length === 0) {
      return null;
    }
    return this.history[this.history.length - 1];
  }
}
