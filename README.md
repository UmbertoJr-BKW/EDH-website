# Hackathon Submission and Leaderboard Platform

This project is a Django-based web application designed to manage submissions for a hackathon. It allows participants to upload their submission files, runs an automated evaluation script, and displays the results on a public leaderboard.

The platform is built with Python and Django, uses Google Cloud Storage for robust file handling, and is deployed on [Render](https://render.com/).

**Live Application:** [https://legs-challenge-bkw.onrender.com](https://legs-challenge-bkw.onrender.com)

## Features

- **User Authentication:** Participants can sign up and log in to manage their submissions.
- **Secure File Uploads:** Submissions, consisting of 7 `.parquet` files, are uploaded directly to Google Cloud Storage.
- **Automated Evaluation:** A custom management command processes pending submissions, calculates a score, and records the results.
- **Live Leaderboard:** Displays all evaluated submissions, ranked by score.
- **Personal Submission History:** Users can view the status and results of their own submissions.

## Tech Stack

- **Backend:** Python 3.12 with Django 5.x
- **Database:** PostgreSQL (Production), SQLite (Local Development)
- **File Storage:** Google Cloud Storage (GCS)
- **Data Processing:** Pandas & Pyarrow
- **Deployment:**
    - **PaaS:** Render
    - **WSGI Server:** Gunicorn
    - **Static Files:** Whitenoise

## Project Structure

```
/
├── .env.example        # Example environment variables file
├── hackathon/          # Django project configuration
│   ├── settings.py     # Main project settings
│   ├── urls.py         # Root URL configuration
│   └── wsgi.py         # WSGI entry point
├── submissions/        # Django app for submission logic
│   ├── models.py       # Database models
│   ├── views.py        # View logic
│   ├── urls.py         # App-specific URL patterns
│   ├── templates/      # HTML templates
│   └── management/     # Custom Django commands
│       └── commands/
│           └── evaluate_submissions.py
├── manage.py           # Django's command-line utility
├── requirements.txt    # Python package dependencies
├── build.sh            # Build script for Render
└── README.md           # This file
```

## How It Works

1.  A user logs in and navigates to the `/submit/` page.
2.  The user uploads 7 `.parquet` files via the HTML form.
3.  A `Submission` record is created in the database with a `PENDING` status.
4.  The view manually uploads each file to a unique, user-specific folder in the Google Cloud Storage bucket.
5.  The database record is updated with the paths to the files in GCS.
6.  Periodically, a cron job on Render runs the `evaluate_submissions` command.
7.  This command queries for `PENDING` submissions, downloads the files from GCS, runs the scoring logic, and creates an `EvaluationResult` record.
8.  The `Submission` status is updated to `COMPLETE` or `ERROR`.
9.  The `/leaderboard/` and `/my-submissions/` pages display the latest results.

---

## Local Development Setup

Follow these steps to run the application on your local machine.

### 1. Prerequisites

- Python 3.12+
- `pip` and `venv`
- (Recommended) [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) for local authentication.

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

For local development, authenticate using your own Google account credentials via the `gcloud` CLI.

```bash
gcloud auth login
gcloud auth application-default login
```

### 4. Environment Variables

Create a `.env` file in the project root. You can copy the example file to get started:

```bash
cp .env.example .env
```

Now, open the `.env` file and fill in the values.

```ini
# .env - For Local Development

# General Django Settings
# Generate a new key: python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
SECRET_KEY=a_long_random_string_for_local_use
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost

# Database (SQLite for local)
DATABASE_URL=sqlite:///db.sqlite3

# Google Cloud Storage
GS_BUCKET_NAME=<your-gcs-bucket-name>
GS_PROJECT_ID=<your-gcp-project-id>

# Leave empty if using 'gcloud auth', otherwise set path to your service account JSON key
GOOGLE_APPLICATION_CREDENTIALS=
```

### 5. Database Setup

```bash
# Apply database migrations
python manage.py migrate

# Create a superuser to access the admin panel
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

This application is deployed on Render and configured for continuous deployment from the `main` branch. The setup consists of three main components.

### 1. Render Service Configuration

-   **Web Service:**
    -   **Runtime:** Python 3
    -   **Build Command:** `bash build.sh`
    -   **Start Command:** `gunicorn hackathon.wsgi`
-   **PostgreSQL Database:**
    -   A managed database instance providing the `DATABASE_URL` to the application.
-   **Cron Job:**
    -   **Command:** `python manage.py evaluate_submissions`
    -   **Schedule:** Runs periodically to process pending submissions.

### 2. Environment Variables

The following environment variables are configured on Render for the Web Service and Cron Job.

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| `SECRET_KEY` | A strong, unique secret key for Django security. | `render_generated_secret_string` |
| `DEBUG` | Must be `False` in a production environment. | `False` |
| `ALLOWED_HOSTS`| The domain name of the Render service. | `bkw-leg-challenge.onrender.com` |
| `DATABASE_URL`| The internal connection string from the Render PostgreSQL instance. | `postgres://user:pass@host/...` |
| `GS_BUCKET_NAME`| Name of the Google Cloud Storage bucket. | `my-hackathon-submissions` |
| `GS_PROJECT_ID`| The Google Cloud Project ID. | `gcp-project-12345` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to the service account JSON key, managed by Render's **Secret Files** feature. | `/etc/secrets/gcs-credentials.json` |
| `PYTHON_VERSION`| The specific Python version to use. | `3.12.4` |

### 3. Creating a Superuser on Production

To create an admin user on the live application, use the **Shell** tab on the Render Web Service dashboard:
```bash
python manage.py createsuperuser
```