# 🤖 Agent Pool

> 내부 문서 검색 + 웹 검색을 수행하는 **멀티툴 지능형 에이전트 시스템**  

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-purple.svg)](#)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](#)

---

## 📋 프로젝트 개요 (Project Overview)

Agent Pool은 **LangGraph(StateGraph) 기반 ReAct 에이전트**로, 사용자의 질문을 받아  
`web_search / doc_search / summarize` 도구를 선택·실행하고 최종 응답을 생성합니다.

특히 본 프로젝트는 **A100 환경에서 30B LLM을 LoRA-SFT로 학습한 뒤, `merge_and_unload()`로 단일 fp16 모델로 병합**하여  
런타임에서 LoRA attach 없이 **추론 구조를 단순화(의존성/복잡도 축소)** 한 것이 핵심 설계 포인트입니다.

---

## ✨ 주요 기능 (Key Features)

- **LangGraph(StateGraph) 기반 ReAct Agent** — Thought → Action → Observation → Final 흐름으로 멀티툴 라우팅
- **멀티 도구 실행** — `web_search`(Serper), `doc_search`(내부 문서), `summarize`
- **출력 안정성 강화** — JSON 강제 대신 키워드 기반 ReAct 포맷 적용, 방어적 파싱 + Action 정규화로 라우팅 실패 최소화, 미지원 Action은 안전하게 FINAL fallback
- **30B 모델 학습–병합–추론 아키텍처** — LoRA-SFT 학습 → `merge_and_unload()` → fp16 단일 모델 서빙, 런타임 LoRA attach 제거로 구조 단순화 및 운영 안정성 강화

---

## 🧠 학습 – 병합 – 추론 아키텍처 (Training → Merge → Inference)

### 1) LoRA-SFT 학습 (ReAct Planner 튜닝)
- ReAct 포맷 데이터셋(JSONL) 생성 후, LoRA 어댑터만 학습
- 관련 파일: `llm_sft/make_react_dataset.py`, `llm_sft/train_react_planner.py`

### 2) 단일 모델 병합 (fp16)
- LoRA 어댑터를 base 모델에 병합하여 단일 fp16 모델 생성
- 런타임에서 LoRA attach 없이 로드 가능
- 관련 파일: `merge_lora.py` (`merge_and_unload()`)

### 3) 런타임 추론 (ReAct Agent)
- Planner 추론을 분리하여(`planner_generate`) Agent 루프 안정화
- Action 정규화/방어적 파싱 기반으로 tool routing 실패 최소화
- 관련 파일: `llm/planner_inference.py`, `agent/agent.py`

---

## 🛠️ 기술 스택 (Tech Stack)

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| LLM / Fine-tuning | HuggingFace Transformers, PEFT(LoRA), TRL(SFTTrainer) |
| Agent Framework | LangGraph(StateGraph) |
| UI | Gradio |
| Web Search | Serper API |
| Infra | A100 GPU Server |

---

## 📁 프로젝트 구조 (Project Structure)

```text
AgentPool_Final-main/
├── app.py                          # 실행 엔트리(Gradio/UI 연결)
├── ui/
│   └── gradio_ui.py                # Gradio UI
│
├── agent/
│   └── agent.py                    # LangGraph(StateGraph) ReAct 에이전트 핵심
│
├── llm/
│   ├── model_loader.py             # 모델 로딩(병합 모델 기본)
│   ├── inference.py                # 공용 추론 유틸
│   └── planner_inference.py        # Planner 추론 분리(planner_generate)
│
├── llm_sft/
│   ├── make_react_dataset.py       # ReAct SFT 데이터 생성(JSONL)
│   ├── make_react_dataset2.py      # 데이터 생성 변형/확장
│   ├── train_react_planner.py      # LoRA-SFT 학습 스크립트
│   └── check_env.py                # 환경 점검
│
├── tools/
│   ├── base.py                     # Tool 인터페이스
│   ├── web_search.py               # Serper 기반 웹 검색 도구
│   ├── doc_search.py               # 내부 문서 검색 도구
│   └── summarize.py                # 요약 도구
│
├── embeddings/
│   └── embedding_loader.py         # 임베딩 로드/유틸
│
├── config/
│   ├── config.py                   # 전역 설정(merged 모델 기본, LoRA attach off)
│   └── paths.py                    # 경로 정의(models/merged_react_30b 등)
│
├── merge_lora.py                   # LoRA + base 병합 → fp16 단일 모델 저장
├── test_react_planner_only.py      # Planner 단독 테스트
└── requirement.txt                 # 의존성 목록
```

---

## 🚀 설치 및 실행 (Installation & Run)

### 1️⃣ 사전 요구사항 (Prerequisites)
- Python 3.10+
- (권장) CUDA 환경 + A100/고성능 GPU
- Serper API Key (웹 검색 도구 사용 시)

### 2️⃣ 설치 (Install)
```bash
git clone https://github.com/hansol418/AgentPool_Final.git
cd AgentPool_Final-main

python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

pip install -r requirement.txt
```

### 3️⃣ 환경 변수 (Environment Variables)
```bash
# 웹 검색(Serper)
export SERPER_API_KEY="YOUR_KEY"

# (선택) 병합 모델 경로를 바꾸고 싶을 때 (기본값: models/merged_react_30b)
export MODEL_NAME="models/merged_react_30b"
```

### 4️⃣ 실행 (Run)
```bash
python app.py
```

---

## 🧪 학습/병합 실행 (Optional: Training & Merge)

**1) ReAct SFT 데이터 생성**
```bash
python llm_sft/make_react_dataset.py
```

**2) LoRA-SFT 학습**
```bash
python llm_sft/train_react_planner.py
```

**3) LoRA 병합 (fp16 단일 모델 생성)**
```bash
python merge_lora.py
```

---

## 🐛 설계 포인트 (Design Notes)

### 출력 포맷 이탈로 인한 tool routing 실패
- JSON 강제 대신 키워드 기반 ReAct 포맷 채택
- `agent/agent.py`에서 방어적 파싱 + Action 정규화로 복구 가능하도록 설계

### 런타임 LoRA attach로 인한 복잡도 증가
- `merge_lora.py`의 `merge_and_unload()`로 단일 fp16 모델 생성
- `config/config.py`에서 `USE_REACT_PLANNER_LORA=False`로 런타임 단순화

---

## 👤 담당 역할 (My Contribution)

| 기능 | 기여도 |
|------|--------|
| LangGraph 기반 StateGraph 설계 및 에이전트 오케스트레이션 | 100% |
| 30B 모델에 맞춘 ReAct 포맷 커스터마이징 (키워드 기반 포맷 + 방어적 파싱) | 100% |
| ReAct Planner 추론 구조 분리 및 안정화 | 100% |
| Serper API 기반 웹 검색 도구 직접 구현 | 100% |
| 내부 문서 검색(RAG 형태) + 요약 파이프라인 구성 | 100% |
| Gradio UI 데모 연결 | 100% |

