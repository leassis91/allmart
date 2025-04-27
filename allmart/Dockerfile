FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user and switch to it
RUN useradd -m kedro
USER kedro

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["kedro", "run"]