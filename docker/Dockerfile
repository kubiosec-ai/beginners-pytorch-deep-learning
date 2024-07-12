# Use the PyTorch image as the base image
FROM python:3.12.4-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Upgrade pip to latest version
RUN /bin/sh -c "pip3 install --upgrade pip"

# Install the required Python packages
RUN /bin/sh -c "pip3 install -r /app/requirements.txt"

# Set the default command to run when starting the container
CMD ["bash"]
