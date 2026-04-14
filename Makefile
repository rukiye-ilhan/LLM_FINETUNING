install:
	pip install -r requirements.txt

run-backend:
	python -m uvicorn backend.app.main:app --reload

run-frontend:
	cd frontend && npm run dev

run-full-inference:
	python scripts/run_full_rag_lora_inference.py

build-counsel-gold:
	python scripts/build_counsel_gold.py

prepare-emotion:
	python scripts/prepare_emotion_tone_dataset.py

build-sft:
	python scripts/build_sft_dataset.py

train-lora:
	python scripts/train_lora.py

test-local-model:
	python scripts/test_local_model.py

test-lora:
	python scripts/test_lora_inference.py

index-corpus:
	python -m src.rag.index_corpus

run-rag-pipeline:
	python -m src.pipelines.run_rag_pipeline

test-retrieval:
	python -m src.rag.test_retrieval

test-reranking:
	python -m src.rag.test_reranking

test-context:
	python -m src.rag.test_context_builder_v2

test-prompt:
	python -m src.rag.test_prompt_builder

docker-build:
	docker build -t emparag:latest .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

dvc-init:
	dvc init

dvc-track:
	dvc add data/raw
	dvc add data/processed
	dvc add data/reports