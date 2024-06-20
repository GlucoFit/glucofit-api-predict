# Use the Kaggle Python image as the base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and to enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_CLOUD_PROJECT=capstone-playground-423804

# Set working directory
WORKDIR /app
    
# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Create the required directories
RUN mkdir -p /app/model /app/csv
RUN mkdir /app/src/controllers/temp

# Expose the port the app runs on
EXPOSE 8080

# Run the application using Gunicorn for production
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "src.app:app"]
