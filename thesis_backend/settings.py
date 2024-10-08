"""
Django settings for thesis_backend project.

Generated by 'django-admin startproject' using Django 4.2.13.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
import sys
import os
import subprocess
from datetime import datetime
import logging
import colorlog

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-^ll18n9oucsp6_=xe8uokbp-(+f%c!k$hts1#bs1r5^k1m$&$6"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "file_processing.apps.FileProcessingConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
]

CORS_ALLOWED_ORIGIN_REGEXES = [
    # allow localhost on any port
    r"^http:\/\/localhost:*([0-9]+)?$",
    r"^https:\/\/localhost:*([0-9]+)?$",
    r"^http:\/\/127.0.0.1:*([0-9]+)?$",
    r"^https:\/\/127.0.0.1:*([0-9]+)?$",
]

ROOT_URLCONF = "thesis_backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "thesis_backend.wsgi.application"


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


def get_git_commit_message():
    try:
        commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"]
        ).strip().decode('utf-8')
        return commit_msg[:18].replace(' ', '_').lower()
    except Exception as e:
        print(f"Error fetching commit message: {e}")
        return "no_commit_msg"


def get_formatted_time():
    now = datetime.now()
    return now.strftime("%H.%M.%S_%d.%m.%y")


commit_msg_part = get_git_commit_message()
formatted_time = get_formatted_time()

base_log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(base_log_dir, exist_ok=True)
log_file_name = f"all_logs_{commit_msg_part}_{formatted_time}.log"
log_file_path = os.path.join(base_log_dir, log_file_name)

# Create a colour formatter
colour_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'INFO_RAW': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'ERROR_RAW': 'light_red',
        'CRITICAL': 'bold_red',
    },
    datefmt='%H:%M:%S %d.%m.%y'
)


# Add custom log level for raw std
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.name == 'STDOUT':
            record.levelname = 'INFO_RAW'
        elif record.name == 'STDERR':
            record.levelname = 'ERROR_RAW'
        return super(CustomFormatter, self).format(record)


# Log writers -->
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
formatter = CustomFormatter('[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
                            datefmt='%H:%M:%S %d.%m.%y')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.__stdout__)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(colour_formatter)
# Log writers <--

# Configure the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = [file_handler, console_handler]
# logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('langchain_community.document_loaders.parsers.doc_intelligence').setLevel(logging.ERROR)
# logging.getLogger('httpx').setLevel(logging.WARNING)


class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.buf = []

    def write(self, message):
        # Progressbars for example need a buffer to get final state
        if message.endswith('\n'):
            self.buf.append(message.removesuffix('\n'))
            out = ''.join(self.buf).strip()
            # Leading/trailing new lines, spaces are useless
            if out.strip():
                self.logger.log(self.log_level, out.strip())
            self.buf = []
        else:
            self.buf.append(message)

    def flush(self):
        pass


# Pipe stdout to logging
stdout_logger = logging.getLogger('STDOUT')
stderr_logger = logging.getLogger('STDERR')
sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
            'datefmt': '%H:%M:%S %d.%m.%y'
        },
    },
    # Propagate messages to the root logger
    'loggers': {
        'django': {
            'handlers': [],
            'level': 'INFO',
            'propagate': True,
        },
        'django.request': {
            'handlers': [],
            'level': 'ERROR',
            'propagate': True,
        },
    },
}