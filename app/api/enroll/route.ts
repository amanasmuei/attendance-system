/**
 * Face Enrollment API Endpoint
 * POST /api/enroll - Store encrypted face descriptors
 */

import { NextRequest, NextResponse } from 'next/server';

export interface EnrollmentRequest {
  userId: string;
  encryptedDescriptor: string;
  encryptionIV: string;
  encryptionKey: string;
  sampleCount: number;
  enrolledAt: string;
}

export interface EnrollmentResponse {
  success: boolean;
  userId: string;
  message: string;
  enrollmentId?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: EnrollmentRequest = await request.json();

    // Validate request body
    const { userId, encryptedDescriptor, encryptionIV, encryptionKey, sampleCount, enrolledAt } =
      body;

    if (
      !userId ||
      !encryptedDescriptor ||
      !encryptionIV ||
      !encryptionKey ||
      !sampleCount ||
      !enrolledAt
    ) {
      return NextResponse.json(
        {
          success: false,
          message: 'Missing required fields',
        },
        { status: 400 },
      );
    }

    // TODO: Validate user exists in database
    // const user = await prisma.user.findUnique({ where: { staffId: userId } });
    // if (!user) {
    //   return NextResponse.json(
    //     { success: false, message: 'User not found' },
    //     { status: 404 }
    //   );
    // }

    // TODO: Store in database using Prisma
    // const enrollment = await prisma.userBiometric.create({
    //   data: {
    //     userId: user.id,
    //     embeddingEncrypted: Buffer.from(encryptedDescriptor, 'base64'),
    //     embeddingHash: await hashDescriptor(encryptedDescriptor),
    //     enrolledAt: new Date(enrolledAt),
    //     expiresAt: new Date(Date.now() + 6 * 30 * 24 * 60 * 60 * 1000), // 6 months
    //     consentTimestamp: new Date(),
    //     sampleCount,
    //   },
    // });

    // Mock response for now
    const enrollmentId = `enroll_${Date.now()}_${userId}`;

    console.log('[Enrollment] New enrollment:', {
      userId,
      sampleCount,
      enrollmentId,
    });

    return NextResponse.json(
      {
        success: true,
        userId,
        enrollmentId,
        message: 'Enrollment successful',
      },
      { status: 201 },
    );
  } catch (error) {
    console.error('[Enrollment] Error:', error);

    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 },
    );
  }
}

/**
 * GET /api/enroll?userId=XXX - Check if user is enrolled
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json(
        {
          success: false,
          message: 'userId parameter is required',
        },
        { status: 400 },
      );
    }

    // TODO: Query database
    // const biometric = await prisma.userBiometric.findFirst({
    //   where: {
    //     user: {
    //       staffId: userId
    //     }
    //   },
    //   include: {
    //     user: true
    //   }
    // });

    // Mock response
    const isEnrolled = false; // Set to true for testing

    if (isEnrolled) {
      return NextResponse.json({
        success: true,
        enrolled: true,
        userId,
        enrolledAt: new Date().toISOString(),
      });
    } else {
      return NextResponse.json({
        success: true,
        enrolled: false,
        userId,
      });
    }
  } catch (error) {
    console.error('[Enrollment] Error checking enrollment:', error);

    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 },
    );
  }
}

/**
 * DELETE /api/enroll?userId=XXX - Delete enrollment (withdraw consent)
 */
export async function DELETE(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json(
        {
          success: false,
          message: 'userId parameter is required',
        },
        { status: 400 },
      );
    }

    // TODO: Delete from database
    // await prisma.userBiometric.deleteMany({
    //   where: {
    //     user: {
    //       staffId: userId
    //     }
    //   }
    // });

    console.log('[Enrollment] Deleted enrollment for:', userId);

    return NextResponse.json({
      success: true,
      message: 'Enrollment deleted successfully',
      userId,
    });
  } catch (error) {
    console.error('[Enrollment] Error deleting enrollment:', error);

    return NextResponse.json(
      {
        success: false,
        message: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 },
    );
  }
}
