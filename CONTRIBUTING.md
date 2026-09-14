# Contributing to NagarSeva

Thank you for contributing to NagarSeva! To keep our codebase stable and prevent version conflicts across different development environments (Windows, macOS, Linux), please follow these guidelines.

---

## 🚀 Quickstart Options

You can develop on NagarSeva using any of the 3 approaches below:

### Option A: Full-Stack with Docker (Recommended for Zero-Conflict)

Docker provides an isolated, uniform environment with **PostgreSQL 16**, **Redis 7**, **Spring Boot (Java 17)**, and **React (Node 20)**.

1. **Prerequisites**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. **Setup environment variables**:
   ```bash
   cp .env.example .env
   ```
3. **Start the entire stack**:
   ```bash
   docker compose up --build
   ```
4. **Access the application**:
   - Frontend UI: [http://localhost:5173](http://localhost:5173)
   - Backend API: [http://localhost:8080/api](http://localhost:8080/api)
   - PostgreSQL: `localhost:5432` (db: `nagarsevadb`, user: `nagarseva`, pass: `nagarseva_dev`)
   - Redis: `localhost:6379`

Any changes you make to the frontend or backend source code will automatically trigger live reloads.

---

### Option B: 1-Click Cloud via GitHub Codespaces

If you don't have Docker installed locally:
1. Navigate to the NagarSeva repository on GitHub.
2. Click **Code** -> **Codespaces** -> **Create codespace on main**.
3. A complete browser-based VS Code environment will launch with Java 17, Node 20, Maven, and Docker pre-installed.

---

### Option C: Local Development without Docker

If developing directly on your host machine:

#### 1. Requirements:
- **Java**: OpenJDK 17 or 21 (Temurin / Eclipse recommended).
- **Node.js**: `v20.x` (or `>= 18.0.0 <= 22.x`). Use [nvm](https://github.com/nvm-sh/nvm) / [fnm](https://github.com/Schniz/fnm) and run `nvm use`.
- **Maven**: 3.9+ (or use `./mvnw`).

#### 2. Running the Backend:
```bash
cd Nagar-Seva/backend
# If you don't have PostgreSQL or Redis installed locally,
# the backend automatically falls back to an in-memory H2 database & in-memory rate limiter!
mvn spring-boot:run
```
- API Base: `http://localhost:8080`
- H2 In-Memory DB Console: `http://localhost:8080/h2-console`

#### 3. Running the Frontend:
```bash
cd Nagar-Seva/frontend
npm install
npm run dev
```
- Frontend UI: `http://localhost:5173`

---

## 🌿 Git Collaboration & Conflict Prevention

To prevent merge disputes and broken builds:

### 1. Always Create a Feature Branch
Never commit directly to `main`. Create a descriptive branch:
```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Keep Your Branch Synchronized
Before submitting code, pull the latest changes from `main` using rebase:
```bash
git fetch origin
git rebase origin/main
```

### 3. Normalize Line Endings (CRLF vs LF)
Our repository includes a `.gitattributes` file that normalizes line endings to Unix `LF`.
Ensure your Git client respects this:
```bash
git config --global core.autocrlf input   # On Linux/macOS
git config --global core.autocrlf true    # On Windows
```

### 4. Lockfile Hygiene
Always install dependencies in the frontend using:
```bash
cd Nagar-Seva/frontend
npm install
```
Do **not** commit unauthorized changes to `package-lock.json` unless you are explicitly adding or upgrading dependencies.

---

## 🧪 Pre-Submission Checklist

Before pushing your branch or opening a Pull Request, verify both backend and frontend pass:

1. **Backend Tests**:
   ```bash
   cd Nagar-Seva/backend
   mvn test
   ```
   *All 81 tests must pass (0 failures, 0 errors).*

2. **Frontend Build**:
   ```bash
   cd Nagar-Seva/frontend
   npm run build
   ```
   *The Vite build must finish with 0 errors.*

---

## 🔒 Security & Environment Secrets
- **Never commit `.env` or `serviceAccountKey.json` files**.
- All local configurations should be placed in your local `.env` file (copied from `.env.example`).
