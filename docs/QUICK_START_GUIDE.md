# Quick Start Guide - PWA Face Recognition Attendance System

## Fastest Path to Implementation

### 1. Project Setup (5 minutes)

```bash
# Create project
mkdir attendance-system
cd attendance-system
npm init -y

# Install dependencies
npm install face-api.js @tensorflow/tfjs idb
npm install -D vite vite-plugin-pwa

# Create directory structure
mkdir -p public/{models,icons} src
```

### 2. Download Pre-trained Models (10 minutes)

```bash
# Download face-api.js models
cd public/models

# Option 1: Use CDN links in your app
# Option 2: Download locally
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-shard1
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_expression_model-weights_manifest.json
curl -O https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_expression_model-shard1
```

### 3. Basic HTML Structure (public/index.html)

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#2196F3" />
    <title>Attendance System</title>
    <link rel="manifest" href="/manifest.json" />
    <style>
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      body {
        font-family: system-ui;
        background: #1a1a1a;
        color: #fff;
      }
      .container {
        position: relative;
        width: 100vw;
        height: 100vh;
      }
      video {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
      canvas {
        position: absolute;
        top: 0;
        left: 0;
      }
      .controls {
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
      }
      button {
        padding: 15px 30px;
        margin: 0 10px;
        border: none;
        border-radius: 50px;
        background: #2196f3;
        color: white;
        cursor: pointer;
        font-size: 16px;
      }
      button:hover {
        background: #1976d2;
      }
      #status {
        position: absolute;
        top: 20px;
        left: 20px;
        padding: 15px;
        background: rgba(0, 0, 0, 0.8);
        border-radius: 10px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <video id="video" autoplay playsinline></video>
      <canvas id="overlay"></canvas>

      <div id="status">Loading...</div>

      <div class="controls">
        <button id="capture">Check In</button>
        <button id="stop">Stop</button>
      </div>
    </div>

    <script type="module" src="/src/app.js"></script>
  </body>
</html>
```

### 4. Minimal Working App (src/app.js)

```javascript
import * as faceapi from 'face-api.js';

class AttendanceApp {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('overlay');
    this.status = document.getElementById('status');
  }

  async init() {
    try {
      // 1. Load models
      this.updateStatus('Loading models...');
      await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
      await faceapi.nets.faceExpressionNet.loadFromUri('/models');

      // 2. Start camera
      this.updateStatus('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 1280, height: 720 },
      });
      this.video.srcObject = stream;

      // 3. Wait for video to be ready
      await new Promise((resolve) => {
        this.video.onloadedmetadata = resolve;
      });

      // 4. Match canvas to video
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;

      // 5. Start detection
      this.updateStatus('Ready!');
      this.startDetection();

      // 6. Setup buttons
      this.setupButtons();
    } catch (error) {
      this.updateStatus('Error: ' + error.message);
    }
  }

  startDetection() {
    setInterval(async () => {
      const detections = await faceapi
        .detectAllFaces(this.video, new faceapi.TinyFaceDetectorOptions())
        .withFaceExpressions();

      // Clear canvas
      const ctx = this.canvas.getContext('2d');
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      if (detections.length > 0) {
        // Resize and draw
        const resized = faceapi.resizeResults(detections, {
          width: this.canvas.width,
          height: this.canvas.height,
        });

        faceapi.draw.drawDetections(this.canvas, resized);
        faceapi.draw.drawFaceExpressions(this.canvas, resized);

        // Show dominant emotion
        const emotion = this.getDominantEmotion(detections[0].expressions);
        this.updateStatus(`Detected: ${emotion}`);
      } else {
        this.updateStatus('No face detected');
      }
    }, 100);
  }

  getDominantEmotion(expressions) {
    return Object.entries(expressions).reduce((a, b) => (a[1] > b[1] ? a : b))[0];
  }

  setupButtons() {
    document.getElementById('capture').onclick = () => this.capture();
    document.getElementById('stop').onclick = () => window.location.reload();
  }

  async capture() {
    const canvas = document.createElement('canvas');
    canvas.width = this.video.videoWidth;
    canvas.height = this.video.videoHeight;
    canvas.getContext('2d').drawImage(this.video, 0, 0);

    const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg'));
    console.log('Captured!', blob);
    this.updateStatus('Check-in successful!');
  }

  updateStatus(message) {
    this.status.textContent = message;
  }
}

// Start app
const app = new AttendanceApp();
app.init();
```

### 5. PWA Manifest (public/manifest.json)

```json
{
  "name": "Attendance System",
  "short_name": "Attendance",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#2196F3",
  "icons": [
    {
      "src": "/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

### 6. Service Worker (public/sw.js)

```javascript
const CACHE_NAME = 'attendance-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/src/app.js',
  '/models/tiny_face_detector_model-weights_manifest.json',
  '/models/face_expression_model-weights_manifest.json',
];

// Install
self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(urlsToCache)));
});

// Fetch
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => response || fetch(event.request))
  );
});
```

### 7. Vite Config (vite.config.js)

```javascript
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      manifest: {
        name: 'Attendance System',
        short_name: 'Attendance',
        theme_color: '#2196F3',
      },
    }),
  ],
});
```

### 8. Run the App

```bash
# Development
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## Adding IndexedDB Storage

```javascript
// Add to app.js
import { openDB } from 'idb';

const db = await openDB('AttendanceDB', 1, {
  upgrade(db) {
    db.createObjectStore('attendance', { keyPath: 'id', autoIncrement: true });
  },
});

// Save attendance
async function saveAttendance(data) {
  await db.add('attendance', {
    timestamp: Date.now(),
    emotion: data.emotion,
    imageBlob: data.imageBlob,
  });
}
```

---

## Adding Face Recognition

```javascript
// Load additional models
await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
await faceapi.nets.faceRecognitionNet.loadFromUri('/models');

// Get face descriptor
const detection = await faceapi
  .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
  .withFaceLandmarks()
  .withFaceDescriptor();

const descriptor = detection.descriptor; // 128D Float32Array

// Save for later matching
await db.put('descriptors', {
  userId: 'user123',
  descriptor: Array.from(descriptor),
});

// Match faces
const savedDescriptors = await db.getAll('descriptors');
const labeledDescriptors = savedDescriptors.map(
  (d) => new faceapi.LabeledFaceDescriptors(d.userId, [new Float32Array(d.descriptor)])
);

const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
const match = faceMatcher.findBestMatch(descriptor);

console.log('Matched user:', match.label, 'Distance:', match.distance);
```

---

## Key URLs and Resources

### Official Documentation

- **MDN Web APIs**: https://developer.mozilla.org/en-US/docs/Web/API
- **TensorFlow.js**: https://www.tensorflow.org/js/guide
- **face-api.js**: https://github.com/justadudewhohacks/face-api.js

### Pre-trained Models

- **face-api.js models**: https://github.com/justadudewhohacks/face-api.js/tree/master/weights
- **TensorFlow.js models**: https://www.tensorflow.org/js/models

### Tools

- **Lighthouse**: Chrome DevTools > Lighthouse tab
- **PWA Builder**: https://www.pwabuilder.com/
- **Workbox**: https://developers.google.com/web/tools/workbox

### Browser Support

- **Can I Use**: https://caniuse.com/
- **MDN Browser Compatibility**: Check each API page

---

## Common Issues and Solutions

### Issue 1: Camera permission denied

```javascript
try {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
} catch (error) {
  if (error.name === 'NotAllowedError') {
    alert('Please allow camera access in your browser settings');
  }
}
```

### Issue 2: Models not loading

```javascript
// Check if models exist
const response = await fetch('/models/tiny_face_detector_model-weights_manifest.json');
if (!response.ok) {
  console.error('Models not found! Download them first.');
}
```

### Issue 3: HTTPS required

```bash
# For local development with HTTPS
npm install -D @vitejs/plugin-basic-ssl

# In vite.config.js
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  plugins: [basicSsl()]
});
```

### Issue 4: Memory leaks

```javascript
// Always dispose tensors
import * as tf from '@tensorflow/tfjs';

function checkMemory() {
  console.log('Tensors:', tf.memory().numTensors);
}

// Call periodically
setInterval(checkMemory, 5000);
```

---

## Performance Optimization Quick Wins

1. **Use Tiny Face Detector** (190KB vs 6MB)
2. **Skip frames** (detect every 2-3 frames)
3. **Lower video resolution** (640x480 instead of 1920x1080)
4. **Cache models** in Service Worker
5. **Use WebGL backend** for TensorFlow.js
6. **Dispose tensors** after use

```javascript
// Optimized detection loop
let frameCount = 0;
setInterval(async () => {
  frameCount++;
  if (frameCount % 2 !== 0) return; // Skip every other frame

  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceExpressions();

  // Process detection...
}, 100);
```

---

## Testing Checklist

- [ ] Works on Chrome/Edge
- [ ] Works on Firefox
- [ ] Works on Safari (desktop)
- [ ] Works on mobile Chrome
- [ ] Works on mobile Safari
- [ ] Works offline
- [ ] Service Worker registers
- [ ] Can be installed as PWA
- [ ] Camera permissions work
- [ ] Face detection runs smoothly (>10 FPS)
- [ ] Data saves to IndexedDB
- [ ] HTTPS enabled

---

## Next Steps

1. **Add User Authentication**: Implement login/registration
2. **Implement Face Recognition**: Match detected faces to known users
3. **Add Backend API**: Sync data to server
4. **Improve UI/UX**: Add better loading states and animations
5. **Add Analytics**: Track attendance patterns
6. **Implement GDPR Compliance**: Add consent forms and data export
7. **Add Push Notifications**: Remind users about events
8. **Optimize Performance**: Web Workers for ML inference
9. **Add Tests**: Unit and integration tests
10. **Deploy**: Host on Vercel, Netlify, or your preferred platform

---

## Deployment Commands

```bash
# Build optimized version
npm run build

# Output goes to dist/ folder
# Upload to your hosting provider

# For Vercel
npm install -g vercel
vercel

# For Netlify
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

---

## Estimated Timeline

- **MVP (Minimal Viable Product)**: 1-2 days
- **Full Features**: 1-2 weeks
- **Production Ready**: 3-4 weeks
- **Enterprise Grade**: 2-3 months

---

## Budget Considerations

**Free Tier Options:**

- **Hosting**: Vercel, Netlify (free tier)
- **Database**: Supabase, Firebase (free tier)
- **Storage**: Cloudflare R2, Backblaze B2 (free tier)
- **CDN**: Cloudflare (free)

**Paid Services (if needed):**

- **Backend**: AWS, Google Cloud, Azure (~$20-100/month)
- **Database**: PostgreSQL on AWS RDS (~$15/month)
- **Storage**: AWS S3 (~$5-20/month)
- **Monitoring**: Sentry (~$26/month)

---

This quick start guide provides the fastest path to a working PWA with face recognition. For complete details, refer to the main documentation file: `PWA_FACE_RECOGNITION_DOCUMENTATION.md`
