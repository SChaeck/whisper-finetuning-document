# Whisper Document

## 1. 사전 지식

### 1.1. Whisper
- OpenAI 개발
- ASR 모델 → STT, 언어 감지, 번역 등의 작업 가능
- Transformer (Encoder-Decoder) 아키텍처 기반으로 설계됨.
- 다양한 언어와 방언 지원 + 배경 소음이 복잡한 환경에서도 상대적으로 정확한 결과 제공
- 사이즈별로 오픈소스 공개

| 모델                | 파라미터 | URL |
|---------------------|----------|-----|
| whisper-large-v3    | 1.54B    | [링크](https://huggingface.co/openai/whisper-large-v3) |
| whisper-large-v3-turbo | 809M  | [링크](https://huggingface.co/openai/whisper-large-v3-turbo) |
| whisper-medium      | 754M     | [링크](https://huggingface.co/openai/whisper-medium) |
| whisper-small       | 242M     | [링크](https://huggingface.co/openai/whisper-small) |
| whisper-base        | 72.6M    | [링크](https://huggingface.co/openai/whisper-base) |
| whisper-tiny        | 37.8M    | [링크](https://huggingface.co/openai/whisper-tiny) |

### 1.2. Mel frequency
- 인간의 청각 특성을 반영하여 음성을 분석할 때 사용되는 주파수 척도.
- 음성 신호 처리에서 사람의 청각과 더 유사한 분석을 수행.

### 1.3. Mel frequency bins
- Mel scale로 변환된 주파수 대역을 나눈 구간.
- Mel frequency bins의 수가 많아질수록 더 세밀한 주파수 정보가 포함되지만, 연산 비용이 증가함.
- Whisper 모델이 사용하는 bins:
  - **128 Mel bins**: large-v3, large-v3-turbo
  - **80 Mel bins**: large-v2, large-v1, medium, small, base, tiny

**⚠️ Note:**
사용하는 모델의 Mel bins을 기준으로 데이터 전처리를 다르게 해야 함.
- large-v3의 FeatureExtractor로 전처리한 데이터(128 Mel bins):
  - large-v3-turbo에서 사용 가능.
  - large-v2 또는 medium에서 사용 불가능(채널 에러 발생).

### 1.4. Character Error Rate (CER)
- 두 문장을 철자 기준으로 비교한 에러율 (주로 모델의 평가 지표로 사용됨).
- Word Error Rate (WER)과 비교:
  - **한국어의 특징**:
    - 형태소 기반의 교착어 → 조사와 접사가 존재.
    - 띄어쓰기 규칙이 비교적 유연.
  - WER은 단어 단위로 비교하기 때문에 작은 철자나 조사, 띄어쓰기에 민감하여 과도하게 오류율이 높게 나오는 경향이 있음.
  - CER은 철자 단위로 비교하기 때문에 이러한 변화에 덜 민감하며, 오류를 적절하게 측정 가능.

## 2. 데이터 준비

### 2.1. AIHub
1. [중·노년층 한국어 방언 데이터 (충청도, 전라도, 제주도)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71558)
2. [중·노년층 한국어 방언 데이터 (강원도, 경상도)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71517)
- **총 데이터 수**: 700k

### 2.2. VOTE400
- [설명 및 다운로드](https://ai4robot.github.io/mindslab-etri-vote400/#)
- 용량: 30GB
- 방언: 대구(DG), 경남(GN), 강원(GW), 전남(JN), 서울(SE)

# 3. 데이터 전처리

### 3.1. 기본 전처리

```python
from transformers import WhisperTokenizer, WhisperProcessor

# feature extractor, tokenizer 로드
model_name = "openai/whisper-large-v3"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(
    model_name, language="Korean", task="transcribe"
)

# 전처리 함수 정의
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# 전처리 적용
train_data = train_data.map(
    prepare_dataset,
    remove_columns=["audio", "sentence"],
    num_proc=2,
    desc="Preparing train dataset",
)
val_data = val_data.map(
    prepare_dataset,
    remove_columns=["audio", "sentence"],
    num_proc=2,
    desc="Preparing validation dataset",
)
test_data = test_data.map(
    prepare_dataset,
    remove_columns=["audio", "sentence"],
    num_proc=2,
    desc="Preparing test dataset",
)
```

### 3.2. SpecAugment

- 데이터 증강 기법
    
    ![image.png](Whisper%20Document%201501115c8eb880fdb0fbebd0220f7258/image%201.png)
    
    ![image.png](Whisper%20Document%201501115c8eb880fdb0fbebd0220f7258/image%202.png)
    
- 잡음에도 Robust한 모델 학습 가능
- 참고자료: https://velog.io/@fbdp1202/Data-Augmentation-in-Audio-and-Speech-Feature-Drop-Aspect#filteraugment

```python
import torchaudio

# SpecAugment 정의
time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
frequency_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)

# SpecAugment 적용 함수
def apply_specaugment(features):
    features = torch.tensor(features).unsqueeze(0)
    features = time_masking(features)
    features = frequency_masking(features)
    return features.squeeze(0).numpy()

def apply_specaugment_to_dataset(batch):
    batch["input_features"] = apply_specaugment(batch["input_features"])
    return batch

# SpecAugment가 적용된 데이터셋 생성
train_data = train_data.map(
    apply_specaugment_to_dataset, 
    num_proc=2,
    desc="Augmenting train dataset",
)
val_data = val_data.map(
    apply_specaugment_to_dataset, 
    num_proc=2, 
    desc="Augmenting validation dataset",
)
test_data = test_data.map(
    apply_specaugment_to_dataset,
    num_proc=2, 
    desc="Augmenting test dataset",
)
```

# 4. 학습

### 파인튜닝

```python
# 1. Data collator (학습시 동적 전처리용)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Process input features
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# 2. data collator 초기화
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# 3. 평가 메트릭(CER) 정의
metric = evaluate.load("cer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}
    
# 4. 개선된 학습 인자 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{target_model_name}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=64,
    learning_rate=5e-7,
    warmup_steps=250,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=250,
    eval_steps=250,
    logging_steps=10,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=2,  # 데이터 로딩 성능 향상
)

# 5. 트레이너 초기화 및 학습
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# 6. 프로세서 미리 저장
processor.save_pretrained(training_args.output_dir)

# 7. 학습
trainer.train()
```

### 설정별 학습 결과

- 데이터
    1. AIHub + Vote400 ⇒ **AV**
    2. AIHub + Vote400 + SpecAugment ⇒ **AVS**
    3. AIHub + Vote400 + SpecAugment + General ⇒ **AVSG**
- 모델
    1. openai/whisper-large-v3 ⇒ **L**
    2. openai/whisper-medium ⇒ **M**
    3. openai/whisper-small ⇒ **S**
    4. openai/whisper-base ⇒ **B**
    5. openai/whisper-tiny ⇒ **T**

| (CER) | 전라도 | 제주도 | 충청도 | 경상도 | 강원도 | 대구(DG) | 경남(GN) | 강원(GW) | 전남(JN) | 서울(SE) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **L + AV** | **0.0556** | **0.0632** | **0.0579** | **0.0710** | **0.0476** | **0.0718** | **0.0275** | **0.0139** | **0.0372** | **0.0201** |
| M + AV | 0.1352 | 0.1103 | 0.0753 | 0.1016 | 0.0840 | 0.0455 | 0.0463 | 0.0346 | 0.0397 | 0.0359 |
| S + AV | 0.4263 | 0.2241 | 0.1235 | 0.1777 | 0.1348 | 0.0668 | 0.0729 | 0.0590 | 0.0773 | 0.0675 |
| B + AV | 0.4696 | 0.3454 | 0.2071 | 0.3084 | 0.3076 | 0.1362 | 0.1510 | 0.1254 | 0.1278 | 0.1037 |
| T + AV | 0.5709 | 0.5023 | 0.3020 | 0.4079 | 0.3466 | 0.2173 | 0.2121 | 0.1799 | 0.2085 | 0.1835 |

| (CER) | 전라도 | 제주도 | 충청도 | 경상도 | 강원도 | 대구(DG) | 경남(GN) | 강원(GW) | 전남(JN) | 서울(SE) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L + AV | **0.0556** | **0.0632** | **0.0579** | **0.0710** | **0.0476** | **0.0718** | **0.0275** | **0.0139** | **0.0372** | **0.0201** |
| L + AVS |  |  |  |  |  |  |  |  |  |  |
| L + AVSG |  |  |  |  |  |  |  |  |  |  |
| S + AV | 0.4263 | 0.2241 | 0.1235 | 0.1777 | 0.1348 | 0.0668 | 0.0729 | 0.0590 | 0.0773 | 0.0675 |
| S + AVS | 0.4356 | 0.2066 | 0.1326 | 0.1892 | 0.1350 | 0.0725 | 0.0800 | 0.0671 | 0.7780 | 0.0751 |

# 5. 사용

### 로컬 실행

- whisper.py
    - 전사 데이터와 모델 추론 결과 비교
    - main 부분만 바꾸면 사용가능
    
    ```python
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from datasets import load_dataset
    import os, json
    import evaluate
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    # CER(Character Error Rate) 메트릭 로드
    cer_metric = evaluate.load("cer")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    ft_model = "local path or HuggingFace name"
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ft_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    def whisper(audio_file, input_filename):
        save_dir = "temp"
    
        temp_audio_path = os.path.join(save_dir, input_filename)
    
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file)
    
        try:
            # pipe 함수 호출 및 경로 전달
            result = pipe(temp_audio_path)
        finally:
            # 사용 후 파일 삭제
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
        return result
    
    if __name__ == "__main__":
        file_path = "script"
        for file in os.listdir(file_path):
            with open(os.path.join(file_path, file), "r") as json_file:
                data = json.load(json_file)
    
            mp3_file_name = data["fileName"] + ".wav"
            script = data["script"]["value"]
    
            try:
                with open(os.path.join("data", mp3_file_name), "rb") as mp3:
                    mp3_bytes = mp3.read()
            except:
                continue   
            result = whisper(mp3_bytes)
    
            cer = cer_metric.compute(predictions=[result["text"]], references=[script])
    
            # 결과 출력
            print(f"File: {file}")
            print(f"Reference (script): {script}")
            print(f"Prediction (Whisper): {result['text']}")
            print(f"CER: {cer:.2f}\n")
    
    ```
    

### RestAPI 터널링

- server.py
    - whisper.py에서 정의한 whisper 함수를 사용해 API 호출 처리
    - webm으로 들어오는 파일은 mp3로 변경해서 사용
    
    ```python
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import RedirectResponse
    from fastapi.middleware.cors import CORSMiddleware
    from langserve import add_routes
    from pydantic import BaseModel
    from whisper import whisper
    import ffmpeg, os
    
    app = FastAPI()
    
    # Set all CORS enabled origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # webm -> mp3 파일
    async def convert_webm_to_mp3(input_bytes: bytes, input_filename: str) -> bytes:
        # webm 파일을 임시 파일로 저장
        input_path = f"temp/{input_filename}"
        with open(input_path, "wb") as f:
            f.write(input_bytes)
        
        # 변환된 mp3 파일의 경로 설정
        output_path = input_path.replace(".webm", ".mp3")
        
        # ffmpeg를 사용하여 변환 작업 수행
        ffmpeg.input(input_path).output(output_path).run()
    
        # 변환된 mp3 파일을 바이트로 읽기
        with open(output_path, "rb") as f:
            mp3_bytes = f.read()
    
        # 임시 파일 삭제
        os.remove(input_path)
        os.remove(output_path)
    
        return mp3_bytes
    
    @app.post("/whisper")
    async def chain_start(audio: UploadFile = File(...)):
        try:
            # 업로드된 파일을 바이트로 읽음
            audio_bytes = await audio.read()
            filename = audio.filename
    
            # 파일 확장자가 webm일 경우 mp3로 변환
            if filename.endswith(".webm"):
                audio_bytes = await convert_webm_to_mp3(audio_bytes, filename)
            
            # whisper 함수에 파일 바이트 전달
            result = whisper(audio_bytes, filename)
            return {"result": result}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    
    if __name__ == "__main__":
        import uvicorn
    
        uvicorn.run(app, host="0.0.0.0", port=8001)
    ```
    
- https://ngrok.com/
    - 터널링을 위한 무료 URL 생성
    - 터널링
        
        ```python
        # 터널링 예시 코드 (사용 불가
        ngrok http --domain=sawfish-charming-pangolin.ngrok-free.app 8001
        ```
        
    - API 호출 가능
        
        ```python
        curl -X POST "https://sawfish-charming-pangolin.ngrok-free.app/whisper" \
          -H "Content-Type: multipart/form-data" \
          -F "audio=@audio_test.mp3"
        ```

---

## Reference
- Radford, Alec et al. _Robust Speech Recognition via Large-Scale Weak Supervision._ [arXiv](https://arxiv.org/abs/2212.04356), 2022.

### 작성자

동국대학교 컴퓨터공학전공

정수채

email: jeongsuchae9211@gmail.com
