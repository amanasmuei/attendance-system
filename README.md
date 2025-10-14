# Event Attendance System - PWA with Face Recognition

Progressive Web App for event attendance tracking using face recognition and emotion analysis.

## Features (Phase 1 - Foundation)

- ✅ Next.js 15 with TypeScript and Tailwind CSS
- ✅ Progressive Web App (PWA) capabilities with offline support
- ✅ PostgreSQL database with Prisma ORM
- ✅ IndexedDB for offline-first data storage
- ✅ ESLint, Prettier, and Husky pre-commit hooks
- ✅ Complete database schema for attendance, users, and biometric data

## Getting Started

### Prerequisites

- Node.js 20 LTS or higher
- PostgreSQL 16+ (for database)
- Modern browser with camera access (Chrome 53+, Firefox 36+, Safari 11+)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/amanasmuei/attendance-system.git
cd attendance-system
```

2. **Install dependencies:**

```bash
npm install
```

3. **Set up environment variables:**

```bash
cp .env.example .env.local
```

Edit `.env.local` and add your database connection string:

```env
DATABASE_URL="postgresql://user:password@localhost:5432/attendance"
```

4. **Initialize the database:**

```bash
# Run Prisma migrations
npx prisma migrate dev --name init

# Generate Prisma Client
npx prisma generate
```

5. **Run the development server:**

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the application.

## Project Structure

```
attendance-system/
├── app/                      # Next.js app directory
│   ├── layout.tsx           # Root layout with PWA metadata
│   ├── page.tsx             # Home page
│   └── offline/             # Offline fallback page
├── lib/                     # Utility libraries
│   └── db/
│       └── indexedDB.ts     # IndexedDB schema and helpers
├── prisma/
│   └── schema.prisma        # Database schema
├── public/
│   └── manifest.json        # PWA manifest
├── docs/                    # Documentation
└── README.md
```

## Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint errors
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting
- `npm run type-check` - Run TypeScript type checking

## Database Schema

The application uses Prisma with PostgreSQL. Key models:

- **User** - User accounts with roles (Admin, Organizer, Attendee)
- **UserBiometric** - Encrypted face embeddings and enrollment data
- **Event** - Event information and settings
- **Attendance** - Check-in records with method and emotion data
- **Consent** - GDPR-compliant consent records
- **FaceRecognitionLog** - Audit logs for face recognition attempts

## PWA Features

- **Offline Support**: App works offline after initial load
- **Installable**: Can be installed on desktop and mobile devices
- **Service Worker**: Automatic caching of assets and API responses
- **IndexedDB**: Local database for offline attendance recording

## Technology Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **PWA**: next-pwa, Workbox
- **Database**: PostgreSQL 16+, Prisma ORM
- **Local Storage**: Dexie.js (IndexedDB wrapper)
- **Code Quality**: ESLint, Prettier, Husky, lint-staged

## Development Roadmap

### Phase 1: Foundation ✅ (Current)

- Project setup and configuration
- PWA capabilities
- Database schema
- Local storage setup

### Phase 2: Face Recognition (Upcoming)

- ML model integration (face-api.js)
- Face enrollment flow
- Recognition engine
- Camera components

### Phase 3: Attendance & Emotion

- Emotion detection
- Check-in UI
- Offline sync
- Alternative methods (manual, QR)

### Phase 4: Privacy & Compliance

- GDPR/CCPA consent management
- Data encryption
- Auto-deletion policies
- Privacy controls

### Phase 5: Polish & Launch

- Accessibility (WCAG 2.1 AA)
- Comprehensive testing
- Performance optimization
- Documentation

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run linting and type checking: `npm run lint && npm run type-check`
4. Commit with conventional commits
5. Create a pull request

## License

Private project - All rights reserved

## Related Documentation

- [GitHub Issue #1](https://github.com/amanasmuei/attendance-system/issues/1) - Complete specification
- [Quick Start Guide](docs/QUICK_START_GUIDE.md)
- [API Reference](docs/API_REFERENCE_CHEATSHEET.md)
- [PWA Documentation](docs/PWA_FACE_RECOGNITION_DOCUMENTATION.md)

---

**Phase 1 Status**: Foundation Complete ✅

Next: Begin Phase 2 - Face Recognition Core
