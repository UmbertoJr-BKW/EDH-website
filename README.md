# Hackathon Submission and Visualization Platform

This project is a sophisticated, Django-based web application designed to manage submissions for a multi-objective optimization hackathon. It provides a full suite of tools for participants, including user registration, file submission, a live sortable leaderboard, and an interactive 3D visualization of the solution space.

The platform is built with Python and Django, uses Google Cloud Storage for robust file handling, and is deployed on [Render](https://render.com/).

**Live Application:** [https://legs-challenge-bkw.onrender.com](https://legs-challenge-bkw.onrender.com)

## Scope and Core Purpose

The primary goal of this platform is to provide a seamless and informative experience for hackathon participants. It automates the entire submission-to-leaderboard pipeline, allowing competitors to focus on their solutions while receiving near-instant feedback on their performance across multiple competing objectives.

## Key Features

-   **User Authentication:** A complete authentication system allows participants to sign up, log in, and manage their own submissions securely.
-   **Multi-File Uploads:** Participants can upload their solution, consisting of 9 `.parquet` files (7 required, 2 optional), directly to Google Cloud Storage.
-   **Automated Multi-Objective Evaluation:** A background job automatically processes pending submissions, calculating three distinct scores based on different objective functions.
-   **Interactive Sortable Leaderboard:** The live leaderboard displays all evaluated submissions and can be sorted in real-time by any of the three objective scores, allowing for deep analysis of the results.
-   **Personal Submission History:** Logged-in users can view a detailed history of their own submissions, including evaluation status, scores, and a list of the files provided.
-   **3D Score Visualization:** An interactive 3D scatter plot visualizes the entire solution space, helping participants understand the trade-offs between the three objective functions. It automatically highlights the **Pareto optimal frontier** in red, showcasing the most efficient solutions.

## Tech Stack

-   **Backend:** Python 3.12 with Django 5.x
-   **Frontend:** HTML, CSS, JavaScript with **Plotly.js** for 3D visualization.
-   **Database:** PostgreSQL (Production), SQLite (Local Development)
-   **File Storage:** Google Cloud Storage (GCS)
-   **Data Processing:** Pandas & Pyarrow
-   **Deployment:**
    -   **PaaS:** Render
    -   **WSGI Server:** Gunicorn
    -   **Static Files:** WhiteNoise

## Project Structure

The project is organized into two main Django apps for clear separation of concerns:

```
/
├── .env.example        # Example environment variables
├── build.sh            # Build script for Render deployment
├── hackathon/          # Django project configuration (settings.py, urls.py)
├── core/               # App for core functionality (homepage, auth, static files)
│   ├── static/         # CSS stylesheets and images
│   └── templates/      # Templates for homepage, registration, etc.
├── submissions/        # App for hackathon-specific logic
│   ├── models.py       # Submission and EvaluationResult database models
│   ├── views.py        # Logic for leaderboard, submission, visualization
│   ├── templates/      # Templates for leaderboard, forms, 3D plot
│   └── management/     # Custom Django commands (e.g., evaluate_submissions)
├── manage.py           # Django's command-line utility
└── requirements.txt    # Python package dependencies
```

## How It Works

1.  A participant **registers** an account or **logs in**.
2.  The user navigates to the `/submit/` page and uploads up to 9 `.parquet` files.
3.  A `Submission` record is created in the PostgreSQL database with a `PENDING` status. The files are uploaded directly to a user-specific folder in Google Cloud Storage.
4.  Periodically, a **Cron Job** on Render runs the `evaluate_submissions` command.
5.  This command queries for `PENDING` submissions, securely downloads the associated files from GCS, and runs the scoring logic to calculate **three objective scores**.
6.  An `EvaluationResult` record is created with the three scores, and the `Submission` status is updated to `COMPLETE` or `ERROR`.
7.  The `/leaderboard/` page displays all results, allowing users to sort by any of the three scores.
8.  The `/visualize/` page fetches all scores and uses Plotly.js to render the interactive 3D plot, calculating and highlighting the Pareto frontier.

---

## Local Development Setup

Follow these steps to run the application on your local machine.

### 1. Prerequisites

-   Python 3.12+
-   `pip` and `venv`
-   (Recommended) [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) for easy local authentication.

### 2. Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-name>

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Google Cloud Authentication (Local)

For local development, authenticate with your Google account credentials via the `gcloud` CLI.

```bash
gcloud auth login
gcloud auth application-default login
```

### 4. Environment Variables

Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env
```

Open the `.env` file and fill in the values for your local setup.

```ini
# .env - For Local Development

# General Django Settings
SECRET_KEY=generate_a_new_secret_key_for_local_use
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost

# Database (SQLite is fine for local)
DATABASE_URL=sqlite:///db.sqlite3

# Google Cloud Storage
GS_BUCKET_NAME=<your-gcs-bucket-name>
GS_PROJECT_ID=<your-gcp-project-id>
GOOGLE_APPLICATION_CREDENTIALS=
```

### 5. Database Setup

```bash
# Apply database migrations to create the tables
python manage.py migrate

# Create a superuser to access the admin panel (/admin/)
python manage.py createsuperuser
```

### 6. Running the Application

```bash
# Start the local development server
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`.

---

## Deployment on Render

This application is designed for continuous deployment from the `main` branch on Render. The production environment consists of three interconnected services.

### 1. Render Service Configuration

When setting up your project on Render, you will create three services:

1.  **Web Service (Your Django App)**
    -   **Runtime:** `Python 3`
    -   **Build Command:** `bash build.sh` (This script installs dependencies, collects static files, and runs database migrations).
    -   **Start Command:** `gunicorn hackathon.wsgi`
    -   **Instance Type:** `Starter` or higher is recommended for live events to prevent sleeping.

2.  **PostgreSQL Database**
    -   Create a new managed PostgreSQL instance. Render will provide a `DATABASE_URL` environment variable, which the application uses automatically.
    -   **Instance Type:** `Starter` or higher is recommended for production.

3.  **Cron Job (For Evaluating Submissions)**
    -   **Command:** `python manage.py evaluate_submissions`
    -   **Schedule:** Set to run at a desired interval (e.g., every 5 or 15 minutes).
    -   **Instance Type:** Match the Web Service instance type (e.g., `Starter`) for consistent performance.

### 2. Environment Variables

Configure the following environment variables in the "Environment" tab for both your **Web Service** and **Cron Job**.

| Variable                         | Description                                                                                       | Example Value                                  |
| :------------------------------- | :------------------------------------------------------------------------------------------------ | :--------------------------------------------- |
| `SECRET_KEY`                     | A strong, unique secret key for Django security.                                                  | (Use Render's "Generate" button)               |
| `DEBUG`                          | Must be set to `False` in a production environment.                                               | `False`                                        |
| `ALLOWED_HOSTS`                  | The domain name of your Render service.                                                           | `legs-challenge-bkw.onrender.com`              |
| `DATABASE_URL`                   | The internal connection string from your Render PostgreSQL instance.                              | `postgres://user:pass@host/...` (Provided)     |
| `GS_BUCKET_NAME`                 | Name of your Google Cloud Storage bucket.                                                         | `my-hackathon-submissions`                     |
| `GS_PROJECT_ID`                  | Your Google Cloud Project ID.                                                                     | `gcp-project-12345`                            |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to the service account JSON key file.                                                        | `/etc/secrets/gcs-credentials.json`            |
| `PYTHON_VERSION`                 | The specific Python version to use.                                                               | `3.12.4`                                       |

**Important Note on `GOOGLE_APPLICATION_CREDENTIALS`:**
Use Render's **Secret Files** feature to securely upload your GCS service account JSON key.
1.  In your service's "Environment" tab, scroll down to "Secret Files".
2.  Click "Add Secret File".
3.  **Filename:** Enter `gcs-credentials.json`.
4.  **Contents:** Paste the entire contents of your JSON key file.
5.  Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path `/etc/secrets/gcs-credentials.json`.

### 3. Creating a Superuser on Production

To create an admin user on the live application, use the **"Shell"** tab on your Render Web Service dashboard:

```bash
python manage.py createsuperuser
```