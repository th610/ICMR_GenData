# 외부 데이터셋 조사 결과

## 1. EmpatheticDialogues

### 접근 방법
- **원본**: HuggingFace에서 loading script 방식 (deprecated since Nov 2023)
- **해결**: Parquet 변환 버전 사용
  ```
  https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet
  ```

### 데이터 규모
- **총 utterances (발화)**: 76,673개
- **총 conversations (대화)**: 17,844개
- **평균 대화 길이**: ~4.3 turns/conversation

### 구조
```python
{
  'conv_id': 'hit:0_conv:1',
  'utterance_idx': 1,
  'context': 'sentimental',  # 감정 상황
  'prompt': '...',
  'speaker_idx': 0 or 1,  # 0=seeker, 1=supporter
  'utterance': '...',
  'selfeval': '5|5|5_2|2|5',
  'tags': ''
}
```

### 평가 결과 (100개 대화)
- Normal: 94.9%
- V1 (맥락 없는 조언): 1.0%
- V2 (명령형): 4.0%
- V3-V5: 0%
- **총 위반율: 5.1%**

### 특징
✅ 멀티턴 대화
✅ 공감 중심
✅ 일반인 수준 (비전문가)
✅ 짧은 응답 (우리 데이터와 유사)
✅ 실제 사람 대화

---

## 2. Reddit SWMH (SuicideWatch & Mental Health)

### 접근 제한 ⚠️
**양쪽 다 제한적 접근**:

#### HuggingFace
- URL: https://huggingface.co/datasets/AIMH/SWMH
- 상태: **Gated Dataset** (승인 필요)
- 요구사항:
  - 연락처 정보 제공
  - **기관 이메일 (.edu, .ac 등) 필수**
  - .com 이메일 거부
  - 5개 조건 동의 필요

#### Zenodo
- URL: https://zenodo.org/doi/10.5281/zenodo.6476179
- 상태: **Restricted Access**
- 요구사항:
  - 기관 이메일로 접근 요청
  - 5개 사용 조건 동의
  - 연구 목적만 사용
  - 개인 식별 시도 금지
  - 데이터 재배포 금지

### 데이터 규모
- **총 posts**: 54,412개
- **출처**: Reddit 정신건강 관련 subreddits
  - SuicideWatch
  - depression
  - anxiety
  - bipolar
  - 기타 (논문 Table 4 참조)

### 구조
- Format: CSV
- Size: 10K-100K
- License: CC-BY-NC-4.0
- Train/Val/Test split 제공

### 특징
✅ 고위험 상황 (V4/V5 테스트용)
✅ 자살 위험 언급
✅ 정신건강 문제
❌ **접근 제한** (기관 이메일 + 승인 필요)
❌ 대화 형식 아님 (단일 post)

---

## 비교 요약

| Dataset | 접근성 | 대화형 | 위반율 | 고위험 | 용도 |
|---------|--------|--------|--------|--------|------|
| **CounselChat** | ✅ 공개 | ❌ 1턴 | 0% | ❌ | 전문가 상한 |
| **EmpatheticDialogues** | ✅ Parquet | ✅ 멀티턴 | 5.1% | ❌ | 일반 대화 |
| **Reddit SWMH** | ⚠️ 제한 | ❌ 단일 | ? | ✅ | 고위험 |

---

## 권장 사항

### 즉시 사용 가능 ✅
1. **EmpatheticDialogues** (17,844 대화)
   - 일반 공감 대화 벤치마크
   - V1/V2 검출 능력 테스트
   - 에이전트 실전 테스트

### 접근 요청 필요 ⚠️
2. **Reddit SWMH** (54,412 posts)
   - 기관 이메일로 요청
   - V4/V5 고위험 테스트용
   - 승인까지 시간 소요

### 대안 고려 💡
3. **다른 공개 Reddit 데이터**
   - r/depression 공개 크롤링 데이터
   - Pushshift API 이용
   - 단, 윤리적 문제 검토 필요

---

## 다음 단계

### Option A: EmpatheticDialogues 활용 (즉시 가능)
```python
# 에이전트 평가
- 17,844개 대화로 대규모 테스트
- 실제 사람 응답 vs 에이전트 응답 비교
- V1/V2 검출 정확도 측정

# 결과 예상
- Normal: 80-90%
- V1/V2: 5-10%
- 우리 합성 데이터보다 현실적
```

### Option B: SWMH 접근 요청 (시간 필요)
```
1. 기관 이메일 준비 (.edu, .ac 등)
2. HuggingFace 또는 Zenodo에 요청
3. 5개 조건 동의
4. 승인 대기 (수일~수주)
5. 고위험 상황 테스트
```

### Option C: 양쪽 병행
- EmpatheticDialogues로 일반 대화 검증 (지금)
- SWMH 접근 요청 제출 (백그라운드)
- 승인되면 고위험 테스트 추가
