FROM python:3.12-slim 

# Set the working directory
WORKDIR /app

# Install system dependencies required to build some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean

# Copy the application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
