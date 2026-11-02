.PHONY: run test docker-build docker-run clean

run:
	streamlit run app.py

test:
	pytest tests/ -v

docker-build:
	docker build -t auto-eda:v6 .

docker-run:
	docker run -p 8501:8501 auto-eda:v6

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete