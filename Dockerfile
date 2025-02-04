# Use the official Python 3.11.11 image
FROM python:3.11.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the main entry point
CMD ["python", "-m", "Example.rag_example"]