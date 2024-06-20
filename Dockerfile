# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and to enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_CLOUD_PROJECT=capstone-playground-423804

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    python3-venv \
    apt-utils

# Create and activate virtual environment
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

RUN mkdir -p /app/model /app/csv

COPY csv/Dummy_data_no_user.csv /app/csv
COPY model/Recommended_Sugar_Intake_Model_TF.h5 /app/model

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["flask", "--app", "appRSI", "run", "--host=0.0.0.0", "--port=8080"]
