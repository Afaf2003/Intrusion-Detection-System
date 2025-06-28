FROM python:3.12-slim

# Install necessary tools and libraries for compiling dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files
COPY requirements.txt /app/
COPY README.md /app/
COPY setup.py /app/
COPY ./src /app/src
COPY ./artifacts /app/artifacts
COPY ./dataset /app/dataset

# Set the PYTHONPATH to include the /app directory
ENV PYTHONPATH=/app 

# Upgrade pip, setuptools, and wheel, then install dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Expose a port for monitoring (TensorBoard or other tools)
EXPOSE 6006

# Set the default command to run your application
CMD ["python", "src/components/data_ingestion.py"]