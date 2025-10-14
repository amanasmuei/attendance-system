/**
 * Camera Access and Video Stream Utilities
 * Handles getUserMedia, camera enumeration, and stream management
 */

export interface CameraConstraints {
  facingMode?: 'user' | 'environment';
  width?: { min?: number; ideal?: number; max?: number };
  height?: { min?: number; ideal?: number; max?: number };
  frameRate?: { ideal?: number; max?: number };
  deviceId?: { exact: string };
}

export interface CameraDevice {
  deviceId: string;
  label: string;
  groupId: string;
}

export class CameraError extends Error {
  constructor(
    message: string,
    public code: string
  ) {
    super(message);
    this.name = 'CameraError';
  }
}

export class CameraManager {
  private stream: MediaStream | null = null;
  private currentDeviceId: string | null = null;

  /**
   * Get list of available video input devices
   */
  async getAvailableDevices(): Promise<CameraDevice[]> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter((device) => device.kind === 'videoinput')
        .map((device) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${device.deviceId.substring(0, 8)}`,
          groupId: device.groupId,
        }));
    } catch (error) {
      throw new CameraError('Failed to enumerate devices', 'ENUMERATE_FAILED');
    }
  }

  /**
   * Check if camera access is supported
   */
  isCameraSupported(): boolean {
    return !!(
      navigator.mediaDevices && navigator.mediaDevices.getUserMedia
    );
  }

  /**
   * Start camera with specified constraints
   */
  async startCamera(constraints: CameraConstraints = {}): Promise<MediaStream> {
    if (!this.isCameraSupported()) {
      throw new CameraError('Camera not supported in this browser', 'NOT_SUPPORTED');
    }

    try {
      // Default constraints
      const defaultConstraints: MediaStreamConstraints = {
        video: {
          facingMode: constraints.facingMode || 'user',
          width: constraints.width || { ideal: 1280 },
          height: constraints.height || { ideal: 720 },
          frameRate: constraints.frameRate || { ideal: 30, max: 60 },
        },
        audio: false,
      };

      // Add specific device if provided
      if (constraints.deviceId) {
        (defaultConstraints.video as MediaTrackConstraints).deviceId = constraints.deviceId;
      }

      this.stream = await navigator.mediaDevices.getUserMedia(defaultConstraints);

      // Store current device ID
      const videoTrack = this.stream.getVideoTracks()[0];
      if (videoTrack) {
        this.currentDeviceId = videoTrack.getSettings().deviceId || null;
      }

      return this.stream;
    } catch (error: unknown) {
      this.handleCameraError(error);
      throw error; // TypeScript won't reach here but needed for type safety
    }
  }

  /**
   * Switch between front and rear cameras
   */
  async switchCamera(currentFacingMode: 'user' | 'environment'): Promise<MediaStream> {
    const newFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';

    // Stop current stream
    this.stopCamera();

    // Start with new facing mode
    return this.startCamera({ facingMode: newFacingMode });
  }

  /**
   * Select a specific camera device
   */
  async selectDevice(deviceId: string): Promise<MediaStream> {
    // Stop current stream
    this.stopCamera();

    // Start with specific device
    return this.startCamera({ deviceId: { exact: deviceId } });
  }

  /**
   * Stop camera and release resources
   */
  stopCamera(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => {
        track.stop();
      });
      this.stream = null;
      this.currentDeviceId = null;
    }
  }

  /**
   * Get current video stream
   */
  getCurrentStream(): MediaStream | null {
    return this.stream;
  }

  /**
   * Get current device ID
   */
  getCurrentDeviceId(): string | null {
    return this.currentDeviceId;
  }

  /**
   * Capture image from video element
   */
  captureImage(videoElement: HTMLVideoElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;

        const context = canvas.getContext('2d');
        if (!context) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        context.drawImage(videoElement, 0, 0);

        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error('Failed to create blob'));
            }
          },
          'image/jpeg',
          0.95
        );
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Get image data from video for ML processing
   */
  getImageData(videoElement: HTMLVideoElement): ImageData {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Failed to get canvas context');
    }

    context.drawImage(videoElement, 0, 0);
    return context.getImageData(0, 0, canvas.width, canvas.height);
  }

  /**
   * Handle camera errors with user-friendly messages
   */
  private handleCameraError(error: unknown): never {
    if (error instanceof DOMException) {
      switch (error.name) {
        case 'NotAllowedError':
        case 'PermissionDeniedError':
          throw new CameraError(
            'Camera permission denied. Please allow camera access in browser settings.',
            'PERMISSION_DENIED'
          );
        case 'NotFoundError':
        case 'DevicesNotFoundError':
          throw new CameraError('No camera found on this device.', 'DEVICE_NOT_FOUND');
        case 'NotReadableError':
        case 'TrackStartError':
          throw new CameraError(
            'Camera is already in use by another application.',
            'DEVICE_IN_USE'
          );
        case 'OverconstrainedError':
          throw new CameraError(
            'Camera does not support the requested constraints.',
            'CONSTRAINTS_NOT_SATISFIED'
          );
        case 'SecurityError':
          throw new CameraError(
            'Camera access blocked by security policy. HTTPS required.',
            'SECURITY_ERROR'
          );
        case 'AbortError':
          throw new CameraError('Camera access aborted.', 'ABORTED');
        default:
          throw new CameraError(`Camera error: ${error.message}`, 'UNKNOWN_ERROR');
      }
    }
    throw new CameraError('Unknown camera error occurred', 'UNKNOWN_ERROR');
  }
}

// Singleton instance for easy access
let cameraManagerInstance: CameraManager | null = null;

export function getCameraManager(): CameraManager {
  if (!cameraManagerInstance) {
    cameraManagerInstance = new CameraManager();
  }
  return cameraManagerInstance;
}

/**
 * Utility function to initialize video element with stream
 */
export async function initializeVideoElement(
  videoElement: HTMLVideoElement,
  stream: MediaStream
): Promise<void> {
  return new Promise((resolve, reject) => {
    videoElement.srcObject = stream;

    videoElement.onloadedmetadata = () => {
      videoElement
        .play()
        .then(() => resolve())
        .catch(reject);
    };

    videoElement.onerror = () => {
      reject(new Error('Failed to load video'));
    };
  });
}
