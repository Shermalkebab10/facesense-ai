FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y     libglib2.0-0     libsm6     libxext6     libxrender-dev     libgomp1     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads static/results

EXPOSE 7860

CMD ["python", "app.py"]
