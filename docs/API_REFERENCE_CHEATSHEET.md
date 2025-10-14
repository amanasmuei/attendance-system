# API Reference Cheatsheet - PWA Face Recognition

## Quick API References

### 1. Service Worker API

```javascript
// REGISTRATION
navigator.serviceWorker
  .register('/sw.js')
  .then((reg) => console.log('SW registered:', reg))
  .catch((err) => console.error('SW error:', err));

// CHECK IF REGISTERED
navigator.serviceWorker.ready.then((reg) => {
  console.log('SW is active');
});

// UPDATE SERVICE WORKER
registration.update();

// UNREGISTER
registration.unregister();
```

**Inside Service Worker (sw.js):**

```javascript
// INSTALL EVENT
self.addEventListener('install', (event) => {
  event.waitUntil(caches.open('v1').then((cache) => cache.addAll(['/index.html', '/app.js'])));
  self.skipWaiting(); // Activate immediately
});

// ACTIVATE EVENT
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== 'v1').map((key) => caches.delete(key)))
      )
  );
  self.clients.claim(); // Take control immediately
});

// FETCH EVENT
self.addEventListener('fetch', (event) => {
  event.respondWith(caches.match(event.request).then((res) => res || fetch(event.request)));
});

// MESSAGE EVENT
self.addEventListener('message', (event) => {
  if (event.data === 'skipWaiting') {
    self.skipWaiting();
  }
});

// BACKGROUND SYNC
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-data') {
    event.waitUntil(syncData());
  }
});

// PUSH NOTIFICATION
self.addEventListener('push', (event) => {
  const data = event.data.json();
  event.waitUntil(self.registration.showNotification(data.title, data.options));
});
```

---

### 2. Cache API

```javascript
// OPEN CACHE
const cache = await caches.open('my-cache');

// ADD SINGLE ITEM
await cache.add('/image.jpg');

// ADD MULTIPLE ITEMS
await cache.addAll(['/index.html', '/app.js', '/style.css']);

// PUT REQUEST/RESPONSE
await cache.put(request, response);

// MATCH REQUEST
const response = await cache.match(request);

// GET ALL REQUESTS
const requests = await cache.keys();

// DELETE ITEM
await cache.delete(request);

// CHECK ALL CACHES
const cacheNames = await caches.keys();

// DELETE CACHE
await caches.delete('old-cache');

// MATCH ACROSS ALL CACHES
const response = await caches.match(request);
```

**Caching Strategies:**

```javascript
// 1. CACHE FIRST
async function cacheFirst(request) {
  const cached = await caches.match(request);
  return cached || fetch(request);
}

// 2. NETWORK FIRST
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    const cache = await caches.open('dynamic');
    cache.put(request, response.clone());
    return response;
  } catch {
    return caches.match(request);
  }
}

// 3. STALE WHILE REVALIDATE
async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);

  const fetchPromise = fetch(request).then((response) => {
    caches.open('dynamic').then((cache) => {
      cache.put(request, response.clone());
    });
    return response;
  });

  return cached || fetchPromise;
}

// 4. NETWORK ONLY
async function networkOnly(request) {
  return fetch(request);
}

// 5. CACHE ONLY
async function cacheOnly(request) {
  return caches.match(request);
}
```

---

### 3. getUserMedia / MediaStream API

```javascript
// BASIC VIDEO CAPTURE
const stream = await navigator.mediaDevices.getUserMedia({
  video: true,
  audio: false,
});

// WITH CONSTRAINTS
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { min: 640, ideal: 1280, max: 1920 },
    height: { min: 480, ideal: 720, max: 1080 },
    aspectRatio: 16 / 9,
    facingMode: 'user', // or 'environment'
    frameRate: { ideal: 30, max: 60 },
  },
});

// SPECIFIC DEVICE
const stream = await navigator.mediaDevices.getUserMedia({
  video: { deviceId: { exact: 'device-id-here' } },
});

// ENUMERATE DEVICES
const devices = await navigator.mediaDevices.enumerateDevices();
const cameras = devices.filter((d) => d.kind === 'videoinput');

// ATTACH TO VIDEO ELEMENT
const video = document.getElementById('video');
video.srcObject = stream;
await video.play();

// GET TRACKS
const videoTrack = stream.getVideoTracks()[0];
const audioTrack = stream.getAudioTracks()[0];

// TRACK CAPABILITIES
const capabilities = videoTrack.getCapabilities();
// { width: {min: 640, max: 1920}, height: {...}, ... }

// TRACK SETTINGS
const settings = videoTrack.getSettings();
// { width: 1280, height: 720, frameRate: 30, ... }

// TRACK CONSTRAINTS
const constraints = videoTrack.getConstraints();

// APPLY CONSTRAINTS
await videoTrack.applyConstraints({
  width: 1920,
  height: 1080,
});

// STOP TRACK
videoTrack.stop();

// STOP ALL TRACKS
stream.getTracks().forEach((track) => track.stop());
```

**Capture Image from Video:**

```javascript
// METHOD 1: Canvas
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
canvas.getContext('2d').drawImage(video, 0, 0);

// As Data URL
const dataUrl = canvas.toDataURL('image/jpeg', 0.95);

// As Blob
const blob = await new Promise((resolve) => {
  canvas.toBlob(resolve, 'image/jpeg', 0.95);
});

// METHOD 2: ImageCapture API
const track = stream.getVideoTracks()[0];
const imageCapture = new ImageCapture(track);
const blob = await imageCapture.takePhoto();
```

---

### 4. IndexedDB API

```javascript
// OPEN DATABASE
const db = await new Promise((resolve, reject) => {
  const request = indexedDB.open('MyDB', 1);
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);

  request.onupgradeneeded = (event) => {
    const db = event.target.result;

    // Create object store
    const store = db.createObjectStore('users', {
      keyPath: 'id',
      autoIncrement: true,
    });

    // Create indexes
    store.createIndex('email', 'email', { unique: true });
    store.createIndex('name', 'name', { unique: false });
  };
});

// ADD/PUT
const tx = db.transaction('users', 'readwrite');
const store = tx.objectStore('users');
await store.put({ id: 1, name: 'John', email: 'john@example.com' });
await tx.complete;

// GET
const tx = db.transaction('users', 'readonly');
const store = tx.objectStore('users');
const user = await store.get(1);

// GET ALL
const users = await store.getAll();

// GET BY INDEX
const index = store.index('email');
const user = await index.get('john@example.com');

// GET ALL BY INDEX
const users = await index.getAll('John');

// DELETE
const tx = db.transaction('users', 'readwrite');
await tx.objectStore('users').delete(1);

// CLEAR
await tx.objectStore('users').clear();

// CURSOR (iterate)
const tx = db.transaction('users', 'readonly');
const store = tx.objectStore('users');
const cursor = await store.openCursor();

while (cursor) {
  console.log(cursor.key, cursor.value);
  cursor = await cursor.continue();
}

// COUNT
const count = await store.count();

// USING idb LIBRARY (SIMPLIFIED)
import { openDB } from 'idb';

const db = await openDB('MyDB', 1, {
  upgrade(db) {
    db.createObjectStore('users', { keyPath: 'id' });
  },
});

await db.put('users', { id: 1, name: 'John' });
const user = await db.get('users', 1);
const users = await db.getAll('users');
await db.delete('users', 1);
```

---

### 5. Web Crypto API

```javascript
// GENERATE KEY
const key = await crypto.subtle.generateKey(
  { name: 'AES-GCM', length: 256 },
  true, // extractable
  ['encrypt', 'decrypt']
);

// ENCRYPT
const iv = crypto.getRandomValues(new Uint8Array(12));
const encrypted = await crypto.subtle.encrypt(
  { name: 'AES-GCM', iv: iv },
  key,
  data // ArrayBuffer
);

// DECRYPT
const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: iv }, key, encrypted);

// EXPORT KEY
const exported = await crypto.subtle.exportKey('jwk', key);

// IMPORT KEY
const key = await crypto.subtle.importKey('jwk', keyData, { name: 'AES-GCM', length: 256 }, true, [
  'encrypt',
  'decrypt',
]);

// HASH
const hash = await crypto.subtle.digest(
  'SHA-256',
  data // ArrayBuffer
);

// RANDOM VALUES
const random = crypto.getRandomValues(new Uint8Array(16));

// GENERATE RSA KEY PAIR
const keyPair = await crypto.subtle.generateKey(
  {
    name: 'RSA-OAEP',
    modulusLength: 2048,
    publicExponent: new Uint8Array([1, 0, 1]),
    hash: 'SHA-256',
  },
  true,
  ['encrypt', 'decrypt']
);

// SIGN (HMAC)
const key = await crypto.subtle.generateKey({ name: 'HMAC', hash: 'SHA-256' }, true, [
  'sign',
  'verify',
]);

const signature = await crypto.subtle.sign('HMAC', key, data);

// VERIFY
const valid = await crypto.subtle.verify('HMAC', key, signature, data);
```

---

### 6. Notification API

```javascript
// REQUEST PERMISSION
const permission = await Notification.requestPermission();
// 'granted', 'denied', or 'default'

// SHOW NOTIFICATION
if (permission === 'granted') {
  new Notification('Title', {
    body: 'Notification body text',
    icon: '/icon.png',
    badge: '/badge.png',
    image: '/image.png',
    data: { custom: 'data' },
    tag: 'unique-tag', // Replaces existing with same tag
    requireInteraction: false,
    silent: false,
    vibrate: [200, 100, 200],
    timestamp: Date.now(),
    actions: [
      { action: 'view', title: 'View', icon: '/view.png' },
      { action: 'close', title: 'Close', icon: '/close.png' },
    ],
  });
}

// NOTIFICATION EVENTS
notification.onclick = (event) => {
  console.log('Clicked');
};

notification.onclose = (event) => {
  console.log('Closed');
};

notification.onerror = (event) => {
  console.log('Error');
};

// SERVICE WORKER NOTIFICATIONS
self.registration.showNotification('Title', options);

// HANDLE NOTIFICATION CLICK IN SW
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'view') {
    event.waitUntil(clients.openWindow('/view'));
  }
});
```

---

### 7. Push API

```javascript
// SUBSCRIBE TO PUSH
const registration = await navigator.serviceWorker.ready;

const subscription = await registration.pushManager.subscribe({
  userVisibleOnly: true,
  applicationServerKey: 'YOUR_VAPID_PUBLIC_KEY',
});

// GET SUBSCRIPTION
const subscription = await registration.pushManager.getSubscription();

// UNSUBSCRIBE
await subscription.unsubscribe();

// SEND SUBSCRIPTION TO SERVER
await fetch('/api/subscribe', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(subscription),
});

// HANDLE PUSH IN SERVICE WORKER
self.addEventListener('push', (event) => {
  const data = event.data.json();

  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: data.icon,
      data: data.data,
    })
  );
});
```

---

### 8. TensorFlow.js API

```javascript
// IMPORT
import * as tf from '@tensorflow/tfjs';

// SET BACKEND
await tf.setBackend('webgl'); // or 'wasm', 'cpu'
await tf.ready();

// LOAD MODEL
const model = await tf.loadGraphModel('/model/model.json');
const model = await tf.loadLayersModel('/model/model.json');

// CREATE TENSOR
const tensor = tf.tensor([1, 2, 3, 4], [2, 2]);
const tensor = tf.zeros([2, 2]);
const tensor = tf.ones([2, 2]);
const tensor = tf.randomNormal([2, 2]);

// FROM PIXELS
const tensor = tf.browser.fromPixels(imageElement);

// TO PIXELS
await tf.browser.toPixels(tensor, canvas);

// OPERATIONS
const result = tensor.add(2);
const result = tensor.mul(3);
const result = tensor.div(2);
const result = tf.matMul(a, b);

// PREDICT
const output = model.predict(input);

// GET DATA
const data = await tensor.data(); // TypedArray
const array = await tensor.array(); // JavaScript array

// DISPOSE
tensor.dispose();
tf.dispose([tensor1, tensor2]);

// MEMORY MANAGEMENT
const info = tf.memory();
// { numTensors: 10, numDataBuffers: 10, numBytes: 4000 }

// TIDY (auto-dispose)
const result = tf.tidy(() => {
  const a = tf.tensor([1, 2, 3]);
  const b = tf.tensor([4, 5, 6]);
  return a.add(b);
}); // a and b are disposed, result is kept

// SCOPE
tf.engine().startScope();
// ... tensor operations
tf.engine().endScope();
```

---

### 9. face-api.js API

```javascript
// IMPORT
import * as faceapi from 'face-api.js';

// LOAD MODELS
await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
await faceapi.nets.faceExpressionNet.loadFromUri('/models');
await faceapi.nets.ageGenderNet.loadFromUri('/models');

// DETECT SINGLE FACE
const detection = await faceapi.detectSingleFace(input, new faceapi.TinyFaceDetectorOptions());

// DETECT ALL FACES
const detections = await faceapi.detectAllFaces(input, new faceapi.SsdMobilenetv1Options());

// WITH LANDMARKS
const detection = await faceapi.detectSingleFace(input).withFaceLandmarks();

// WITH DESCRIPTORS
const detection = await faceapi.detectSingleFace(input).withFaceLandmarks().withFaceDescriptor();

// WITH EXPRESSIONS
const detection = await faceapi.detectSingleFace(input).withFaceExpressions();

// WITH AGE AND GENDER
const detection = await faceapi.detectSingleFace(input).withAgeAndGender();

// EVERYTHING
const detection = await faceapi
  .detectSingleFace(input, new faceapi.TinyFaceDetectorOptions())
  .withFaceLandmarks()
  .withFaceDescriptor()
  .withFaceExpressions()
  .withAgeAndGender();

// DETECTOR OPTIONS
new faceapi.TinyFaceDetectorOptions({
  inputSize: 416, // 128, 160, 224, 320, 416, 512, 608
  scoreThreshold: 0.5,
});

new faceapi.SsdMobilenetv1Options({
  minConfidence: 0.5,
  maxResults: 10,
});

// FACE MATCHING
const descriptors = [
  new faceapi.LabeledFaceDescriptors('person1', [descriptor1]),
  new faceapi.LabeledFaceDescriptors('person2', [descriptor2]),
];

const faceMatcher = new faceapi.FaceMatcher(descriptors, 0.6);
const match = faceMatcher.findBestMatch(newDescriptor);
// { label: 'person1', distance: 0.35 }

// DRAW ON CANVAS
const displaySize = { width: 640, height: 480 };
faceapi.matchDimensions(canvas, displaySize);

const resizedDetections = faceapi.resizeResults(detections, displaySize);

faceapi.draw.drawDetections(canvas, resizedDetections);
faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

// CUSTOM DRAW
const box = detection.detection.box;
const drawBox = new faceapi.draw.DrawBox(box, { label: 'Face' });
drawBox.draw(canvas);
```

---

## Browser Compatibility Table (2025)

| Feature         | Chrome  | Firefox | Safari   | Edge    | Mobile       |
| --------------- | ------- | ------- | -------- | ------- | ------------ |
| Service Workers | 40+ ✅  | 44+ ✅  | 11.1+ ✅ | 17+ ✅  | iOS 11.3+ ✅ |
| Cache API       | 43+ ✅  | 41+ ✅  | 11.1+ ✅ | 79+ ✅  | iOS 11.3+ ✅ |
| getUserMedia    | 53+ ✅  | 36+ ✅  | 11+ ✅   | 79+ ✅  | iOS 11+ ⚠️   |
| IndexedDB       | 24+ ✅  | 16+ ✅  | 10+ ✅   | 79+ ✅  | iOS 10+ ✅   |
| Web Crypto      | 37+ ✅  | 34+ ✅  | 11+ ✅   | 79+ ✅  | iOS 11+ ✅   |
| Notifications   | 22+ ✅  | 22+ ✅  | 16.4+ ⚠️ | 79+ ✅  | iOS 16.4+ ⚠️ |
| Push API        | 50+ ✅  | 44+ ✅  | 16.4+ ⚠️ | 79+ ✅  | Android ✅   |
| WebGL           | 9+ ✅   | 4+ ✅   | 5.1+ ✅  | 79+ ✅  | iOS 8+ ✅    |
| WebAssembly     | 57+ ✅  | 52+ ✅  | 11+ ✅   | 79+ ✅  | iOS 11+ ✅   |
| MediaRecorder   | 47+ ✅  | 25+ ✅  | 14.1+ ✅ | 79+ ✅  | iOS 14.3+ ✅ |
| ImageCapture    | 59+ ✅  | ❌      | ❌       | 79+ ✅  | Android ✅   |
| WebGPU          | 113+ 🧪 | 🚧      | ❌       | 113+ 🧪 | ❌           |

Legend: ✅ Full Support | ⚠️ Partial Support | ❌ No Support | 🧪 Experimental | 🚧 In Development

---

## Common Patterns

### Pattern 1: Offline-First with Sync

```javascript
// Save to IndexedDB
async function saveData(data) {
  await db.put('store', { ...data, synced: false });

  if (navigator.onLine) {
    await syncToServer(data);
  } else {
    await registerBackgroundSync();
  }
}

// Background sync
async function registerBackgroundSync() {
  const registration = await navigator.serviceWorker.ready;
  await registration.sync.register('sync-data');
}

// In service worker
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-data') {
    event.waitUntil(syncAllData());
  }
});
```

### Pattern 2: Progressive Enhancement

```javascript
if ('serviceWorker' in navigator) {
  // Register service worker
} else {
  console.log('Service Worker not supported');
}

if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
  // Use camera
} else {
  // Show file upload fallback
}
```

### Pattern 3: Optimistic UI Updates

```javascript
// Update UI immediately
updateUI(data);

// Save locally
await db.put('store', data);

// Sync to server in background
syncToServer(data).catch((error) => {
  // Revert UI on error
  revertUI(data);
});
```

---

## Error Handling

```javascript
// Camera errors
try {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
} catch (error) {
  switch (error.name) {
    case 'NotAllowedError':
      console.error('Permission denied');
      break;
    case 'NotFoundError':
      console.error('No camera found');
      break;
    case 'NotReadableError':
      console.error('Camera in use');
      break;
    case 'OverconstrainedError':
      console.error('Constraints not satisfied');
      break;
    case 'SecurityError':
      console.error('Security error');
      break;
  }
}

// IndexedDB errors
try {
  await db.put('store', data);
} catch (error) {
  if (error.name === 'QuotaExceededError') {
    console.error('Storage quota exceeded');
  }
}

// Service Worker errors
navigator.serviceWorker.register('/sw.js').catch((error) => {
  console.error('SW registration failed:', error);
});
```

---

## Performance Tips

1. **Lazy load models**: Load only when needed
2. **Use Web Workers**: Offload ML inference
3. **Dispose tensors**: Prevent memory leaks
4. **Cache aggressively**: Cache all static assets
5. **Optimize images**: Compress before storage
6. **Use IndexedDB**: For large datasets
7. **Batch operations**: Process multiple frames together
8. **Skip frames**: Don't process every frame
9. **Lower resolution**: Use 640x480 for detection
10. **Monitor memory**: Check tf.memory() regularly

---

## Security Best Practices

1. **HTTPS only**: All PWA features require HTTPS
2. **Encrypt sensitive data**: Use Web Crypto API
3. **Request permissions**: Always ask before accessing camera
4. **Validate input**: Sanitize all user input
5. **Secure backend**: Use authentication tokens
6. **CORS headers**: Configure properly
7. **CSP headers**: Implement Content Security Policy
8. **Data retention**: Auto-delete old data
9. **User consent**: GDPR compliance
10. **Audit regularly**: Security reviews

---

## Debugging Commands

```javascript
// Check service worker status
navigator.serviceWorker.getRegistrations().then(console.log);

// Check cache contents
caches.keys().then(console.log);
caches
  .open('v1')
  .then((cache) => cache.keys())
  .then(console.log);

// Check IndexedDB
indexedDB.databases().then(console.log);

// Check TensorFlow.js memory
tf.memory();

// Check backend
tf.getBackend();

// List all tensors
tf.engine().state.tensorInfo;

// Network status
navigator.onLine;

// Storage estimate
navigator.storage.estimate().then(console.log);
```

---

For complete implementation details, see:

- **Main Documentation**: PWA_FACE_RECOGNITION_DOCUMENTATION.md
- **Quick Start**: QUICK_START_GUIDE.md
