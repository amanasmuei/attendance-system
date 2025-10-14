/**
 * IndexedDB Schema for Event Attendance System
 * Provides offline-first data storage for face recognition and attendance tracking
 */

import Dexie, { type EntityTable } from 'dexie';

// Interface definitions
export interface User {
  id: string;
  userId: string;
  name: string;
  email: string;
  staffId: string;
  faceDescriptor: number[]; // 128-dimensional face embedding
  enrolledAt: number;
  consentGiven: boolean;
  expiresAt: number; // Unix timestamp
}

export interface Attendance {
  id: string;
  userId: string;
  eventId: string;
  timestamp: number;
  method: 'face' | 'manual' | 'qr';
  faceConfidence?: number;
  emotion?: {
    type: string;
    confidence: number;
  };
  synced: boolean;
}

export interface Consent {
  id: string;
  userId: string;
  timestamp: number;
  consentVersion: string;
  purpose: string;
  retentionPeriod: string;
  acknowledged: boolean;
}

export interface Event {
  id: string;
  eventId: string;
  name: string;
  date: number;
  location: string;
  settings: {
    requireFaceRecognition: boolean;
    trackEmotions: boolean;
  };
}

// Database class
class AttendanceDB extends Dexie {
  users!: EntityTable<User, 'id'>;
  attendance!: EntityTable<Attendance, 'id'>;
  consents!: EntityTable<Consent, 'id'>;
  events!: EntityTable<Event, 'id'>;

  constructor() {
    super('AttendanceDB');

    this.version(1).stores({
      users: '++id, userId, staffId, enrolledAt, expiresAt',
      attendance: '++id, userId, eventId, timestamp, synced',
      consents: '++id, userId, timestamp',
      events: '++id, eventId, date',
    });
  }
}

// Export singleton instance
export const db = new AttendanceDB();

// Helper functions
export async function enrollUser(user: Omit<User, 'id'>) {
  return await db.users.add(user as User);
}

export async function getEnrolledFaces() {
  const users = await db.users.toArray();
  return users.map((u) => ({
    label: u.userId,
    descriptors: [new Float32Array(u.faceDescriptor)],
  }));
}

export async function recordAttendance(attendance: Omit<Attendance, 'id'>) {
  return await db.attendance.add(attendance as Attendance);
}

export async function getUnsyncedAttendance() {
  return await db.attendance.where('synced').equals(0).toArray();
}

export async function markAttendanceSynced(ids: string[]) {
  return await db.attendance.bulkUpdate(
    ids.map((id) => ({
      key: id,
      changes: { synced: true },
    }))
  );
}

export async function deleteExpiredUsers() {
  const now = Date.now();
  return await db.users.where('expiresAt').below(now).delete();
}

// Face Recognition specific helpers

/**
 * Get user by face descriptor match
 */
export async function getUserByFaceDescriptor(
  descriptor: Float32Array,
  threshold = 0.6
): Promise<User | null> {
  const users = await db.users.toArray();

  for (const user of users) {
    const userDescriptor = new Float32Array(user.faceDescriptor);
    const distance = euclideanDistance(descriptor, userDescriptor);

    if (distance < threshold) {
      return user;
    }
  }

  return null;
}

/**
 * Get all enrolled face descriptors for matching
 */
export async function getAllFaceDescriptors(): Promise<
  Array<{ userId: string; userName: string; descriptor: Float32Array; enrolledAt: number }>
> {
  const users = await db.users.toArray();
  return users.map((user) => ({
    userId: user.userId,
    userName: user.name,
    descriptor: new Float32Array(user.faceDescriptor),
    enrolledAt: user.enrolledAt,
  }));
}

/**
 * Update user's face descriptor
 */
export async function updateFaceDescriptor(userId: string, descriptor: Float32Array) {
  return await db.users
    .where('userId')
    .equals(userId)
    .modify({ faceDescriptor: Array.from(descriptor) });
}

/**
 * Delete user and their face data
 */
export async function deleteUserFaceData(userId: string) {
  return await db.users.where('userId').equals(userId).delete();
}

/**
 * Calculate Euclidean distance between two face descriptors
 */
function euclideanDistance(descriptor1: Float32Array, descriptor2: Float32Array): number {
  if (descriptor1.length !== descriptor2.length) {
    throw new Error('Descriptors must have same length');
  }

  let sum = 0;
  for (let i = 0; i < descriptor1.length; i++) {
    const diff = descriptor1[i] - descriptor2[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}
