/**
 * ML Model Loader Service
 * Handles progressive loading and caching of face-api.js models
 */

import * as faceapi from '@vladmandic/face-api';

export type ModelType = 'detection' | 'landmarks' | 'recognition' | 'expression' | 'all';

interface ModelLoadState {
  detection: boolean;
  landmarks: boolean;
  recognition: boolean;
  expression: boolean;
}

class ModelLoader {
  private modelPath = '/models';
  private loadState: ModelLoadState = {
    detection: false,
    landmarks: false,
    recognition: false,
    expression: false,
  };
  private loadingPromise: Promise<void> | null = null;

  /**
   * Check if specific models are loaded
   */
  public isLoaded(modelType: ModelType = 'all'): boolean {
    if (modelType === 'all') {
      return Object.values(this.loadState).every((loaded) => loaded);
    }
    return this.loadState[modelType];
  }

  /**
   * Get current load state
   */
  public getLoadState(): ModelLoadState {
    return { ...this.loadState };
  }

  /**
   * Load face detection model (SSD MobileNetV1)
   */
  private async loadDetectionModel(): Promise<void> {
    if (this.loadState.detection) return;

    console.log('[ModelLoader] Loading SSD MobileNetV1 detection model...');
    await faceapi.nets.ssdMobilenetv1.loadFromUri(this.modelPath);
    this.loadState.detection = true;
    console.log('[ModelLoader] Detection model loaded successfully');
  }

  /**
   * Load face landmark model (68 points)
   */
  private async loadLandmarksModel(): Promise<void> {
    if (this.loadState.landmarks) return;

    console.log('[ModelLoader] Loading face landmark 68 model...');
    await faceapi.nets.faceLandmark68Net.loadFromUri(this.modelPath);
    this.loadState.landmarks = true;
    console.log('[ModelLoader] Landmarks model loaded successfully');
  }

  /**
   * Load face recognition model (128-dimensional embeddings)
   */
  private async loadRecognitionModel(): Promise<void> {
    if (this.loadState.recognition) return;

    console.log('[ModelLoader] Loading face recognition model...');
    await faceapi.nets.faceRecognitionNet.loadFromUri(this.modelPath);
    this.loadState.recognition = true;
    console.log('[ModelLoader] Recognition model loaded successfully');
  }

  /**
   * Load face expression model (7 emotions)
   */
  private async loadExpressionModel(): Promise<void> {
    if (this.loadState.expression) return;

    console.log('[ModelLoader] Loading face expression model...');
    await faceapi.nets.faceExpressionNet.loadFromUri(this.modelPath);
    this.loadState.expression = true;
    console.log('[ModelLoader] Expression model loaded successfully');
  }

  /**
   * Load specific models
   */
  public async loadModels(modelType: ModelType = 'all'): Promise<void> {
    // If already loading, wait for that to complete
    if (this.loadingPromise) {
      return this.loadingPromise;
    }

    // Create loading promise
    this.loadingPromise = this.performLoad(modelType);

    try {
      await this.loadingPromise;
    } finally {
      this.loadingPromise = null;
    }
  }

  /**
   * Perform the actual model loading
   */
  private async performLoad(modelType: ModelType): Promise<void> {
    const startTime = performance.now();

    try {
      switch (modelType) {
        case 'detection':
          await this.loadDetectionModel();
          break;

        case 'landmarks':
          await this.loadLandmarksModel();
          break;

        case 'recognition':
          await this.loadDetectionModel();
          await this.loadLandmarksModel();
          await this.loadRecognitionModel();
          break;

        case 'expression':
          await this.loadDetectionModel();
          await this.loadLandmarksModel();
          await this.loadExpressionModel();
          break;

        case 'all':
          await Promise.all([
            this.loadDetectionModel(),
            this.loadLandmarksModel(),
            this.loadRecognitionModel(),
            this.loadExpressionModel(),
          ]);
          break;

        default:
          throw new Error(`Unknown model type: ${modelType}`);
      }

      const endTime = performance.now();
      const duration = ((endTime - startTime) / 1000).toFixed(2);
      console.log(`[ModelLoader] Models loaded in ${duration}s`);
    } catch (error) {
      console.error('[ModelLoader] Error loading models:', error);
      throw new Error(`Failed to load ${modelType} models: ${error}`);
    }
  }

  /**
   * Unload all models (for testing/cleanup)
   */
  public unloadModels(): void {
    this.loadState = {
      detection: false,
      landmarks: false,
      recognition: false,
      expression: false,
    };
    console.log('[ModelLoader] Models unloaded');
  }

  /**
   * Progressive loading: Load tiny models first, then full models
   */
  public async loadProgressively(): Promise<void> {
    console.log('[ModelLoader] Starting progressive loading...');

    // Load tiny detection model first for quick feedback
    await faceapi.nets.tinyFaceDetector.loadFromUri(this.modelPath);
    console.log('[ModelLoader] Tiny detector loaded (quick preview available)');

    // Then load full models
    await this.loadModels('all');
  }
}

// Export singleton instance
export const modelLoader = new ModelLoader();

// Export types and class
export { ModelLoader };
