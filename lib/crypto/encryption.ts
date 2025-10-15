/**
 * Web Crypto API Encryption Utilities
 * AES-256-GCM encryption for biometric data
 */

/**
 * Generate a random encryption key
 */
export async function generateKey(): Promise<CryptoKey> {
  return await window.crypto.subtle.generateKey(
    {
      name: 'AES-GCM',
      length: 256,
    },
    true, // extractable
    ['encrypt', 'decrypt'],
  );
}

/**
 * Derive encryption key from password using PBKDF2
 */
export async function deriveKeyFromPassword(
  password: string,
  salt: Uint8Array,
  iterations: number = 100000,
): Promise<CryptoKey> {
  // Import password as key material
  const keyMaterial = await window.crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey'],
  );

  // Derive key using PBKDF2
  return await window.crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt as BufferSource,
      iterations,
      hash: 'SHA-256',
    },
    keyMaterial,
    {
      name: 'AES-GCM',
      length: 256,
    },
    true,
    ['encrypt', 'decrypt'],
  );
}

/**
 * Generate random salt
 */
export function generateSalt(length: number = 16): Uint8Array {
  return window.crypto.getRandomValues(new Uint8Array(length));
}

/**
 * Generate random IV (Initialization Vector)
 */
export function generateIV(length: number = 12): Uint8Array {
  return window.crypto.getRandomValues(new Uint8Array(length));
}

/**
 * Encrypt data using AES-256-GCM
 */
export async function encrypt(
  data: ArrayBuffer,
  key: CryptoKey,
  iv?: Uint8Array,
): Promise<{
  encrypted: ArrayBuffer;
  iv: Uint8Array;
}> {
  const _iv = iv || generateIV();

  const encrypted = await window.crypto.subtle.encrypt(
    {
      name: 'AES-GCM',
      iv: _iv as BufferSource,
    },
    key,
    data,
  );

  return {
    encrypted,
    iv: _iv,
  };
}

/**
 * Decrypt data using AES-256-GCM
 */
export async function decrypt(encrypted: ArrayBuffer, key: CryptoKey, iv: Uint8Array): Promise<ArrayBuffer> {
  return await window.crypto.subtle.decrypt(
    {
      name: 'AES-GCM',
      iv: iv as BufferSource,
    },
    key,
    encrypted,
  );
}

/**
 * Encrypt string to base64
 */
export async function encryptString(
  plaintext: string,
  key: CryptoKey,
): Promise<{
  encrypted: string;
  iv: string;
}> {
  const data = new TextEncoder().encode(plaintext);
  const { encrypted, iv } = await encrypt(data.buffer as ArrayBuffer, key);

  return {
    encrypted: arrayBufferToBase64(encrypted),
    iv: arrayBufferToBase64(iv.buffer as ArrayBuffer),
  };
}

/**
 * Decrypt base64 string
 */
export async function decryptString(encrypted: string, key: CryptoKey, iv: string): Promise<string> {
  const encryptedData = base64ToArrayBuffer(encrypted);
  const ivData = new Uint8Array(base64ToArrayBuffer(iv));

  const decrypted = await decrypt(encryptedData, key, ivData);
  return new TextDecoder().decode(decrypted);
}

/**
 * Encrypt Float32Array (face descriptor)
 */
export async function encryptDescriptor(
  descriptor: Float32Array,
  key: CryptoKey,
): Promise<{
  encrypted: string;
  iv: string;
}> {
  const { encrypted, iv } = await encrypt(descriptor.buffer as ArrayBuffer, key);

  return {
    encrypted: arrayBufferToBase64(encrypted),
    iv: arrayBufferToBase64(iv.buffer as ArrayBuffer),
  };
}

/**
 * Decrypt to Float32Array (face descriptor)
 */
export async function decryptDescriptor(encrypted: string, key: CryptoKey, iv: string): Promise<Float32Array> {
  const encryptedData = base64ToArrayBuffer(encrypted);
  const ivData = new Uint8Array(base64ToArrayBuffer(iv));

  const decrypted = await decrypt(encryptedData, key, ivData);
  return new Float32Array(decrypted);
}

/**
 * Export key to base64 string
 */
export async function exportKey(key: CryptoKey): Promise<string> {
  const exported = await window.crypto.subtle.exportKey('raw', key);
  return arrayBufferToBase64(exported);
}

/**
 * Import key from base64 string
 */
export async function importKey(keyData: string): Promise<CryptoKey> {
  const buffer = base64ToArrayBuffer(keyData);
  return await window.crypto.subtle.importKey(
    'raw',
    buffer,
    {
      name: 'AES-GCM',
      length: 256,
    },
    true,
    ['encrypt', 'decrypt'],
  );
}

/**
 * Hash data using SHA-256
 */
export async function hash(data: ArrayBuffer): Promise<string> {
  const hashBuffer = await window.crypto.subtle.digest('SHA-256', data as BufferSource);
  return arrayBufferToBase64(hashBuffer);
}

/**
 * Hash string using SHA-256
 */
export async function hashString(text: string): Promise<string> {
  const data = new TextEncoder().encode(text);
  return await hash(data.buffer as ArrayBuffer);
}

/**
 * Hash Float32Array (face descriptor)
 */
export async function hashDescriptor(descriptor: Float32Array): Promise<string> {
  return await hash(descriptor.buffer as ArrayBuffer);
}

/**
 * Convert ArrayBuffer to base64 string
 */
export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Convert base64 string to ArrayBuffer
 */
export function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

/**
 * Convert Uint8Array to hex string
 */
export function uint8ArrayToHex(array: Uint8Array): string {
  return Array.from(array)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Convert hex string to Uint8Array
 */
export function hexToUint8Array(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes;
}

/**
 * Secure random string generator
 */
export function generateSecureRandomString(length: number = 32): string {
  const array = window.crypto.getRandomValues(new Uint8Array(length));
  return arrayBufferToBase64(array.buffer).substring(0, length);
}
