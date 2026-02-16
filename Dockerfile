FROM python:3.11-slim

# Install Chromium and dependencies for headless browser
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    fonts-noto-color-emoji \
    libgbm1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libasound2 \
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
ENV DISPLAY=:99

# Default: run family_with_kids headless
# Override via env vars in Railway:
#   PERSONA_ID=budget_backpacker (single persona)
#   RUN_MODE=batch  PERSONA_ID=family_with_kids,business_traveler (batch)
#   RUN_MODE=batch  (no PERSONA_ID = all personas)
ENV RUN_MODE=single
ENV PERSONA_ID=family_with_kids

CMD ["sh", "-c", "\
  if [ \"$RUN_MODE\" = 'batch' ]; then \
    if [ -n \"$PERSONA_ID\" ]; then \
      python cli.py batch --personas $PERSONA_ID --headless; \
    else \
      python cli.py batch --headless; \
    fi; \
  else \
    python cli.py run $PERSONA_ID --headless; \
  fi"]
