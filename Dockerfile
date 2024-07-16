# Use the official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies in a virtual environment
COPY requirements.txt requirements.txt

# Install virtualenv
RUN pip install --no-cache-dir virtualenv

# Create a virtual environment
RUN virtualenv venv

# Activate the virtual environment and install dependencies
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the environment variable to use the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Expose the port the Flask app runs on
EXPOSE 5000

# Set the entrypoint script as the entry point
ENTRYPOINT ["/app/entrypoint.sh"]
