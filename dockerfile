# Use an official Python runtime as a parent image
FROM python:3.10.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

EXPOSE 5000

# Run final.py when the container launches
CMD ["python", "final.py"]
