
# Hackathon Submission and Leaderboard Platform

This project is a Django-based web application designed to manage submissions for a hackathon. It allows participants to upload their submission files, runs an automated evaluation script, and displays the results on a public leaderboard.

The platform is built with Python and Django and uses Google Cloud Storage for robust and scalable file handling.

## Features

- **User Authentication:** Participants can sign up and log in to manage their submissions.
- **Secure File Uploads:** Submissions, consisting of 7 `.parquet` files, are uploaded directly to Google Cloud Storage.
- **Automated Evaluation:** A custom management command processes pending submissions, calculates a score, and records the results.
- **Live Leaderboard:** Displays all evaluated submissions, ranked by score.
- **Personal Submission History:** Users can view the status and results of their own submissions.

## Tech Stack

- **Backend:** Python 3.12+ with Django 5.x
- **Database:** PostgreSQL (for production), SQLite (for local development)
- **File Storage:** Google Cloud Storage (GCS)
- **Data Processing:** Pandas & Pyarrow (for reading `.parquet` files)
- **Deployment:** Gunicorn (WSGI Server) on a PaaS like Render.

## Project Structure

```
/
├── .env                # Environment variables file
├── hackathon/          # Django project configuration
│   ├── settings.py     # Main project settings
│   ├── urls.py         # Root URL configuration
│   └── wsgi.py         # WSGI entry point
├── submissions/        # Django app for submission logic
│   ├── models.py       # Database models (Submission, EvaluationResult)
│   ├── views.py        # View logic for upload, leaderboard, etc.
│   ├── urls.py         # App-specific URL patterns
│   ├── templates/      # HTML templates
│   └── management/     # Custom Django commands
│       └── commands/
│           └── evaluate_submissions.py # The evaluation script
├── manage.py           # Django's command-line utility
├── requirements.txt    # Python package dependencies
└── README.md           # This file
```

## How It Works

1.  A user logs in and navigates to the `/submit/` page.
2.  The user uploads 7 `.parquet` files via the HTML form.
3.  The `upload_submission` view in `submissions/views.py` receives the files.
4.  A `Submission` record is created in the database with a `PENDING` status.
5.  The view manually uploads each file to a unique, user-specific folder in the Google Cloud Storage bucket.
6.  The database record is updated with the paths to the files in GCS.
7.  Periodically, a cron job runs the `evaluate_submissions` command.
8.  This command queries for `PENDING` submissions, downloads the associated files from GCS, runs the scoring logic, and creates an `EvaluationResult` record.
9.  The `Submission` status is updated to `COMPLETE` or `ERROR`.
10. The `/leaderboard/` and `/my-submissions/` pages query the database to display the latest results.

---

## Local Development Setup

Follow these steps to run the application on your local machine for development and testing.

### 1. Prerequisites

- Python 3.12+
- `pip` and `venv`
- (Optional but Recommended) [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) for local authentication.

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

For local development, you can authenticate using your own Google account credentials via the `gcloud` CLI.

```bash
# Log in with your Google account
gcloud auth login

# Set up Application Default Credentials
gcloud auth application-default login
```

### 4. Environment Variables

The application uses a `.env` file to manage environment variables for local development.

```bash
# Create a .env file from the example
cp .env.example .env
```

Now, open the `.env` file and fill in the values:

```ini
# .env - For Local Development

# General Settings
SECRET_KEY=a_long_random_string_for_local_use
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost

# Database (SQLite for local)
DATABASE_URL=sqlite:///db.sqlite3

# Google Cloud Storage
GS_BUCKET_NAME=<your-gcs-bucket-name>
GS_PROJECT_ID=<your-gcp-project-id>

# Leave this empty if using gcloud auth, otherwise set the path to your JSON key
GOOGLE_APPLICATION_CREDENTIALS=
```

### 5. Database Setup

```bash
# Apply database migrations to create the tables
python manage.py migrate

# Create a superuser account to log into the admin panel
python manage.py createsuperuser
```

### 6. Running the Application

```bash
# Start the local development server
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`. You can log in to the admin panel at `/admin/`.

---

## Production Deployment (Render)

This application is configured for easy deployment to a Platform-as-a-Service (PaaS) like Render.

### 1. Prerequisites

- A Render account (or similar PaaS).
- A PostgreSQL database created on the platform.
- A Google Cloud Service Account with a JSON key file. The service account needs the **`Storage Object Admin`** role on your GCS bucket.

### 2. Push to GitHub

Ensure your latest code is pushed to a GitHub repository that your PaaS account can access. **Make sure your `.env` file and `gcs-credentials.json` are listed in `.gitignore` and are NOT committed.**

### 3. Configure the Web Service on Render

- **Create a New Web Service** and connect it to your GitHub repository.
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn hackathon.wsgi`
- **Environment Variables:**
  - `SECRET_KEY`: Generate a new, strong secret key.
  - `DEBUG`: `False`
  - `ALLOWED_HOSTS`: `your-app-name.onrender.com`
  - `DATABASE_URL`: Use the internal connection string provided by your Render PostgreSQL database.
  - `GS_BUCKET_NAME`: `<your-gcs-bucket-name>`
  - `GS_PROJECT_ID`: `<your-gcp-project-id>`
  - `GOOGLE_APPLICATION_CREDENTIALS`: Use Render's "Secret File" feature. Upload your service account JSON key and set this variable to the path provided by Render (e.g., `/etc/secrets/your-key-file-name`).

### 4. Configure the Cron Job on Render

- **Create a New Cron Job** connected to the same repository.
- **Schedule:** Set your desired evaluation frequency (e.g., every 15 minutes: `*/15 * * * *`).
- **Command:** `python manage.py evaluate_submissions`
- **Environment Variables:** Link the same environment group used by your web service so the cron job can access the database and GCS.