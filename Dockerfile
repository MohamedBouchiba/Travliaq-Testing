FROM python:3.11-slim

# Install browser dependencies (Playwright provides its own Chromium + SwiftShader)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation \
    fonts-noto-color-emoji \
    libgbm1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libasound2 \
    libvulkan1 \
    mesa-vulkan-drivers \
    libegl1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium
RUN python -m playwright install chromium

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p output/results output/screenshots output/conversations

# Set environment for headless Chromium
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
ENV DISABLE_GPU=1

# Expose dashboard port
EXPOSE 8099

# Start dashboard with auto-batch (headless, no browser auto-open)
CMD ["python", "cli.py", "dashboard", "--run-batch", "--headless", "--no-browser"]
