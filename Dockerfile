# Use official Python image
FROM --platform=linux/amd64 python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY app.py .
COPY pneumonia_model.h5 .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run Flask server
CMD ["python", "app.py"]