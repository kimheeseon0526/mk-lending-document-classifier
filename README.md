# MK Lending Document Classifier

셔플된 모기지 대출 서류 패키지 PDF의 각 페이지를 5종 문서 유형으로 분류하고,
물리적으로 연속된 페이지 그룹과 논리적으로 이어지는 원본 문서를 재구성한다.
룰 기반 1차 분류에 저신뢰 페이지만 선택적으로 LLM에 위임하는 하이브리드
구조를 사용하며, LLM 위임 경로는 실제 네트워크 호출까지 구현해 두 패키지
모두에서 실행을 확인했다.

## 핵심 결과

| 항목 | 결과 |
|---|---|
| 분류 방식 | 룰 기반 1차 분류 + 저신뢰 페이지 선택적 LLM 위임 (실제 호출 구현) |
| 출력 | 페이지별 5종 문서 유형 + 물리 그룹 + 논리 문서 재조립 |
| package_01 | rule-only accuracy **0.9744 (38/39)**, macro_f1_supported **0.9853**\* / hybrid accuracy **0.9744 (38/39)**, macro_f1_supported **1.0000**\* |
| package_02 | 정답지 부재로 예측 분포와 실행 지표만 보고 (rule-only·hybrid 모두) |
| 테스트 | **102 passed, 1 warning**, 원본 PDF 없이 핵심 로직 재현 가능 |
| LLM 구현 범위 | 위임 판단·프롬프트·응답 검증·재시도/fallback·실제 네트워크 호출까지 구현. 제출 시점 실행은 Gemini의 OpenAI 호환 엔드포인트(`gemini-2.5-flash`) 사용, 실제 OpenAI 엔드포인트로는 호출하지 않음 |

> \* package_01의 `macro_f1_supported`는 support가 1 이상인 클래스 중 F1이
> 정의되는 클래스만 평균한 프로젝트 정의 지표이며, 고정 5클래스 macro-F1과
> 다르다. hybrid의 1.0000은 성능 향상이 아니라 `INCOME_DOC`이 한 건도 예측되지
> 않아 F1이 정의되지 않고 평균에서 제외된 결과다. 상세는
> [평가 방법 및 결과](#6-평가-방법-및-결과) 참고.

## 목차

1. [개요](#1-개요)
2. [문서 유형](#2-문서-유형)
3. [아키텍처](#3-아키텍처)
4. [실행 방법](#4-실행-방법)
5. [출력 형식](#5-출력-형식)
6. [평가 방법 및 결과](#6-평가-방법-및-결과)
7. [그룹핑](#7-그룹핑)
8. [오답 분석](#8-오답-분석)
9. [AI 사용](#9-ai-사용)
10. [보안 및 데이터 정책](#10-보안-및-데이터-정책)
11. [선택하지 않은 방식](#11-선택하지-않은-방식)
12. [한계 및 개선 방향](#12-한계-및-개선-방향)

## 1. 개요

MK Lending Corp의 중장기 목표는 자체 AUS(Automated Underwriting System)
구축이다. AUS가 서류를 근거로 판정하려면 정확한 데이터 추출이 선행돼야 하는데,
실무 대출 패키지는 수십~수백 페이지가 순서 없이 하나로 합쳐진 PDF로 들어오기
때문에 그 전에 "이 페이지가 무슨 서류의 몇 번째 장인가"부터 판별하는 문서
분류·분할 단계가 필요하다. 이 프로젝트는 그 단계를 구현한다: 셔플된 PDF 한
개를 입력으로 받아 물리 페이지마다 5개 라벨 중 하나를 예측하고, 페이지 마커와
문서 식별자를 근거로 원본 문서 단위를 재구성한다. 룰 기반 분류를 1차로
사용하고, 룰이 확신하지 못하는 페이지만 LLM에 위임하는 하이브리드 구조를
채택했다 — 근거는 3절 참고. `data/package_01`을 대상으로 룰 전용(rule-only)
모드를 실행했을 때 accuracy 0.9744(38/39)를 확인했고, 위임된 페이지에 실제
LLM을 호출하는 hybrid 모드로도 동일 데이터를 실행해 결과를 비교했다(6·9절).

## 2. 문서 유형

| 라벨 | 설명 |
|---|---|
| `URLA_1003` | 대출 신청서 (Uniform Residential Loan Application) |
| `INCOME_DOC` | 소득 증빙 서류 (급여명세서, W-2, 손익계산서 등) |
| `CREDIT_REPORT` | 신용조회 보고서 |
| `TITLE_REPORT` | 권원조사 보고서 (Preliminary Title Report) |
| `OTHER` | 위 네 유형에 해당하지 않거나 판정 불가 |

`DocType`(`src/schema.py`)에 정의된 값 그대로이며 임의로 축약하지 않는다.

## 3. 아키텍처

```
extract.py            PyMuPDF로 페이지 텍스트·회전·마커·식별자 추출
  │
rule_classifier.py    키워드 시그니처 스코어링, 위임(should_delegate) 판정
  │
llm_classifier.py     (위임된 페이지만) PII 마스킹 → 프롬프트 구성 → 배치 →
                      응답 검증 → 재시도/Fallback
  │
grouping.py           물리적 연속 그룹핑(필수) + 논리 문서 재조립(확장)
  │
evaluate.py           정답지 대조 (--truth 제공 시)
visualize.py          페이지 스트립·confusion matrix 이미지 (--viz 제공 시)
```

`pipeline.py`가 위 단계를 순서대로 호출하고 `scripts/run.py`가 CLI 진입점이다.

**기술 스택 선택 근거**:

| 선택 | 선택 이유 | 한계 |
|---|---|---|
| Python 3.11 | PDF·데이터 처리 라이브러리가 풍부하고 파이프라인 구현을 빠르게 반복할 수 있다. | 패키징과 의존성 재현성은 별도 관리가 필요하다. |
| PyMuPDF | 페이지 단위 텍스트, 회전값, 이미지 수를 한 라이브러리에서 추출할 수 있다. | 텍스트 레이어가 없는 스캔 PDF에는 OCR이 필요하다. |
| Pydantic v2 | 모듈 경계의 데이터 계약과 LLM 응답 스키마를 동일 모델로 검증할 수 있다. | Structured Outputs 연동은 Gemini의 OpenAI 호환 레이어로만 검증했고, 실제 OpenAI 엔드포인트로는 검증하지 못했다(9절). |
| PyYAML | 5개 문서 유형의 시그니처·가중치·임계값을 분류 코드와 분리할 수 있다. | 새 `DocType` 추가와 설정 스키마 검증에는 Python 변경이 필요하다. |
| pytest | 원본 PDF를 공개하지 않고도 분류·마스킹·fallback 경로를 synthetic fixture로 확인할 수 있다. | 실제 문서 일반화 성능을 대신하지 못한다. |
| matplotlib | 원본 페이지 이미지를 저장하지 않고 라벨 분포와 confusion matrix를 시각화할 수 있다. | 페이지 내용 자체는 확인할 수 없다. |

**룰 + LLM 결합 근거**: package_01에서는 룰 시그니처만으로 39페이지 중
37페이지가 위임 불필요로 판단되고, p8과 p36 2페이지만 LLM 위임 대상으로
탐지됐다(6절). 재현성(같은 입력에 같은 결과)을 유지하면서 애매한 페이지에만
추가 판단을 구할 수 있다. 전 페이지를
LLM에 보내는 방식은 실행마다 결과가 달라질 위험과 비용·속도 부담이 있어 채택하지
않았다 (11절).

## 4. 실행 방법

**환경**: Python 3.11 이상 (3.11.5에서 검증), 의존성은 `requirements.txt`. `.env.example`을 `.env`로
복사하고 `LLM_API_KEY`를 채우면 `LLMRuntimeConfig.enabled`가 `True`로 바뀌고,
`hybrid` 모드에서 위임된 페이지에 대해 실제 네트워크 호출이 수행된다(자세한
동작은 9절 참고). `LLM_BASE_URL`을 함께 설정하면 OpenAI 호환 엔드포인트를 쓸
수 있다 — 비워두면 실제 OpenAI 엔드포인트(`api.openai.com`)를 그대로 사용한다.
제출 시점 실행은 `LLM_BASE_URL`을 Gemini의 OpenAI 호환 엔드포인트
(`https://generativelanguage.googleapis.com/v1beta/openai/`)로, `LLM_MODEL`을
`gemini-2.5-flash`로 설정해 확인했다 — `LLM_API_KEY`가 Gemini AI Studio에서
발급받은 키이기 때문이다. 키가 없으면 `hybrid` 모드가 `LLM_DISABLED`로
rule-only와 동일하게 자동 강등된다.

**주의**: 원본 파일명에 공백과 `&`가 포함되므로 모든 경로 인자를 따옴표로
감싸야 한다.

**셸**: 아래 명령은 bash 기준이며 줄바꿈에 `\`를 사용한다. Windows PowerShell에서는
`\` 대신 백틱(`` ` ``)으로 바꾸거나 명령을 한 줄로 이어서 입력해야 한다. Git Bash나
WSL에서는 그대로 실행된다.

```bash
# 1) 정답지 생성 (해시 매칭, package_01 전용)
python scripts/build_ground_truth.py \
  --shuffled "data/package_01/01.990145627_shuffled.pdf" \
  --source URLA_1003="data/package_01/1003 - URLA_990145627.pdf" \
  --source CREDIT_REPORT="data/package_01/Credit Report_990145627.pdf" \
  --source INCOME_DOC="data/package_01/INCOME - P & L_990145627.pdf" \
  --source TITLE_REPORT="data/package_01/Title Report_990145627.pdf" \
  --out data/ground_truth_package_01.csv

# 2) rule-only 실행 + 평가 + 시각화
python scripts/run.py \
  --pdf "data/package_01/01.990145627_shuffled.pdf" \
  --mode rule-only \
  --out outputs/package_01 \
  --truth data/ground_truth_package_01.csv \
  --viz

# 3) hybrid 실행 (API 키가 있으면 위임된 페이지에 실제 LLM 호출을 수행한다.
#    없으면 자동으로 rule-only와 동일하게 강등된다)
python scripts/run.py \
  --pdf "data/package_01/01.990145627_shuffled.pdf" \
  --mode hybrid \
  --out outputs/package_01_hybrid \
  --truth data/ground_truth_package_01.csv

# 4) 모듈 단위 확인
python -m src.extract "data/package_01/01.990145627_shuffled.pdf"
python -m src.rule_classifier "data/package_01/01.990145627_shuffled.pdf"
python -m src.grouping "data/package_01/01.990145627_shuffled.pdf"
python -m src.llm_classifier "data/package_01/01.990145627_shuffled.pdf"

# 5) 테스트
pytest -v
```

## 5. 출력 형식

`scripts/run.py --out <dir>`이 저장하는 파일:

| 파일 | 내용 |
|---|---|
| `page_results.csv` | 페이지별 최종 라벨. 주요 컬럼: `page_number`, `doc_type`, `decision_source`, `rule_score`, `rule_margin`, `marker_style`, `marker_page`, `marker_total`, `instance_id`, `is_orphan`, `warnings` |
| `physical_groups.csv` | 물리적 연속 그룹. `group_id`, `doc_type`, `start_page`, `end_page`, `page_count`, `physical_pages` |
| `reconstructed_documents.csv` | 재조립 문서. `doc_id`, `doc_type`, `instance_id`, `marker_style`, `expected_pages`, `observed_logical_pages`, `missing_logical_pages`, `duplicate_logical_pages`, `first_page_at`, `last_page_at`, `physical_span`, `is_contiguous`, `completeness`, `issues`, `logical_to_physical` |
| `run_report.json` | 실행 통계 (`RunReport`) — 총 페이지, 룰 최종 출력 페이지 수, LLM 위임 대상/호출/fallback 수, orphan 목록 등 |
| `evaluation.json` / `evaluation.txt` | `--truth` 제공 시 생성. 정확도·클래스별 지표·confusion matrix |
| `page_strip.png` / `confusion_matrix.png` | `--viz` 제공 시 생성 |

모든 산출 파일은 `PageResult` 이하 비식별 모델에서만 만들어지며, 저장 직전
`pipeline.check_no_pii`가 SSN·이메일 패턴을 검사한다(10절).

## 6. 평가 방법 및 결과

**정답지 생성**: `package_01`은 셔플 전 4개 원본 PDF(URLA, Credit Report, P&L,
Title Report)를 함께 제공한다. `scripts/build_ground_truth.py`가 각 페이지의
공백 제거 정규화 텍스트를 SHA-256 해시로 비교해, 셔플된 PDF의 물리 페이지를
원본 파일·원본 페이지 번호에 매칭한다. 페이지 수 일치, 해시 유일성, 전량 매칭,
중복 매칭 없음을 검증한 뒤에만 CSV를 쓴다. 별도 정답지 파일 없이 39페이지
전량을 이 방식으로 라벨링했다.

**지표**: accuracy, 클래스별 precision/recall/F1,
`macro_f1_supported`. `macro_f1_supported`는 `support > 0`인 클래스 중에서도
**F1이 정의되는 클래스만** 평균한 프로젝트 정의 지표다(`src/evaluate.py`의
`supported_f1_values` — F1이 `None`인 클래스는 평균에서 제외). package_01
rule-only에서는 support>0인 4개 클래스(`URLA_1003`, `INCOME_DOC`,
`CREDIT_REPORT`, `TITLE_REPORT`) 전부 F1이 정의돼 그대로 4개를 평균하지만,
아래 hybrid 결과에서는 이 조건이 실제로 클래스 하나를 제외시킨다. `OTHER`는
rule-only·hybrid 모두 support 0이라 recall/F1을 0.0이 아니라 `N/A`로 표기하고
평균에서 제외한다. 일반적인 고정 5클래스 macro-F1과 구분하기 위해 이름을
`macro_f1_supported`로 쓰며, 전체 클래스 macro-F1처럼 표현하지 않는다.

**결과 (rule-only, package_01, 39페이지)**:

```
accuracy: 0.9744 (38/39)
macro_f1_supported: 0.9853

doc_type        support  predicted  precision  recall   f1
URLA_1003       11       11         1.0000     1.0000   1.0000
INCOME_DOC       1        1         1.0000     1.0000   1.0000
CREDIT_REPORT   18       18         1.0000     1.0000   1.0000
TITLE_REPORT     9        8         1.0000     0.8889   0.9412
OTHER            0        1         0.0000     N/A      N/A
```

confusion matrix:

```
actual \ predicted   URLA  INCOME  CREDIT  TITLE  OTHER
URLA_1003             11      0       0       0      0
INCOME_DOC             0      1       0       0      0
CREDIT_REPORT          0      0      18       0      0
TITLE_REPORT           0      0       0       8      1
OTHER                  0      0       0       0      0
```

오분류는 페이지 8 한 건뿐이다(실제 `TITLE_REPORT`, 예측 `OTHER`). 자세한 원인은
8절 참고.

`run_report.json`의 `rule_resolved`는 충분한 신뢰도로 확정된 페이지 수가 아니라,
최종 `decision_source`가 `RULE`인 페이지 수다. rule-only 모드에서는 위임 대상으로
탐지된 페이지도 최종 출력이 룰 결과로 남기 때문에 package_01의 실행 통계는 아래처럼
구분해 읽어야 한다.

| 항목 | 값 |
|---|---|
| 룰 최종 출력 페이지 (`rule_resolved`) | 39 |
| 위임 불필요 페이지 | 37 |
| LLM 위임 대상으로 탐지된 페이지 | 2 (p8, p36) |
| 실제 LLM 호출 페이지 | 0 |

![Page classification strip](outputs/package_01/page_strip.png)

정답 행과 예측 행이 다른 페이지는 p8 한 장이며(`evaluation.txt`의
`misclassified pages`), 해당 칸이 굵은 테두리로 강조 표시된다.

![Confusion matrix](outputs/package_01/confusion_matrix.png)

이 수치는 `package_01` 39페이지 전체를 룰 개발과 평가에 함께 사용한 결과다.
별도 검증 데이터로 측정한 일반화 성능이 아니다(12절 1항).

### package_01 — hybrid 실행 결과

동일한 `ground_truth_package_01.csv`를 대상으로 `--mode hybrid`를 실행했다.
`LLM_API_KEY`(Gemini AI Studio 발급)와 `LLM_BASE_URL`(Gemini의 OpenAI 호환
엔드포인트)을 설정한 상태로, 위임 대상 2페이지(p8, p36) 모두 실제로 LLM을
호출했다 — `outputs/package_01_hybrid/run_report.json`의 `llm_called: 2`,
`rule_fallback_count: 0`으로 확인된다.

```
accuracy: 0.9744 (38/39)
macro_f1_supported: 1.0000

doc_type        support  predicted  precision  recall   f1
URLA_1003       11       11         1.0000     1.0000   1.0000
INCOME_DOC       1        0          N/A       0.0000   N/A
CREDIT_REPORT   18       18         1.0000     1.0000   1.0000
TITLE_REPORT     9        9         1.0000     1.0000   1.0000
OTHER            0        1         0.0000     N/A      N/A
```

rule-only 대비 바뀐 것은 두 가지다.

- **p8(TITLE_REPORT)이 정정됐다.** rule-only의 유일한 오분류였던 p8을 LLM이
  신뢰도 0.95로 `TITLE_REPORT`로 올바르게 판정해, `TITLE_REPORT`의 recall이
  0.8889(8/9) → 1.0000(9/9)으로 개선됐다.
- **p36(INCOME_DOC)이 새로 틀렸다.** rule-only에서는 맞았던 p36을 LLM이
  신뢰도 0.70으로 `OTHER`로 오판했다(8절 참고). 그 결과 `INCOME_DOC`은 한
  번도 예측되지 않아(predicted_count=0) precision이 0/0으로 정의되지 않고,
  precision이 없으면 F1도 정의되지 않는다.

accuracy는 38/39로 rule-only와 동일하다 — 오분류 페이지가 p8에서 p36으로
**교체**됐을 뿐 개수는 그대로다.

`macro_f1_supported`가 0.9853 → 1.0000으로 바뀐 것은 "성능이 좋아져서"가
아니다. rule-only에서는 `INCOME_DOC`도 F1=1.0으로 평균에 포함돼 4개 클래스
`(1.0+1.0+1.0+0.9412)/4 = 0.9853`이었다. hybrid에서는 `INCOME_DOC`의 F1이
정의되지 않아(위 표의 `N/A`) 평균에서 **제외**되고, 남은 3개 클래스가 전부
1.0이라 `(1.0+1.0+1.0)/3 = 1.0000`이 된다. 즉 이 수치 상승은 TITLE_REPORT의
실제 개선과, INCOME_DOC이 평균 대상에서 통계적으로 빠진 것이 동시에 작용한
결과다. "위임이 정답률을 개선했다"는 근거로 쓸 수 없다 — p36 한 페이지만
보면 룰보다 오히려 나빠졌다.

### package_02 (990367284_shuffled)

package_02는 정답지가 없는 추가 입력으로 사용했으며, 최초 실행에서 발견된 서식
커버리지 갭을 분석해 룰을 확장했다(아래 참고). 따라서 package_02는 독립
검증셋이나 일반화 성능의 근거로 사용하지 않는다. **accuracy는 측정할 수 없으므로
보고하지 않는다.** 아래 수치는 정확도가 아니라 예측 라벨 분포, 위임 대상 수,
빈 텍스트 페이지 수 및 재조립 상태를 나타낸다.

| 항목 | 값 |
|---|---|
| 총 페이지 | 44 |
| 룰 최종 출력 페이지 (`rule_resolved`) | 44 |
| 위임 불필요 페이지 | 40 |
| LLM 위임 대상으로 탐지된 페이지 | 4 |
| 실제 LLM 호출 페이지 | 0 |
| 물리적 그룹 | 44 |
| 재조립 문서 | 4 |
| orphan 페이지 | 11 |

| 예측 라벨 | 페이지 수 |
|---|---|
| CREDIT_REPORT | 14 |
| TITLE_REPORT | 11 |
| URLA_1003 | 10 |
| INCOME_DOC | 5 |
| OTHER | 4 |

#### hybrid 실행 결과

package_01과 동일한 설정(Gemini의 OpenAI 호환 엔드포인트, `gemini-2.5-flash`)으로
package_02도 `--mode hybrid`로 실행했다. 위임 대상 4페이지(p11, p26, p32, p40)
모두 실제로 LLM을 호출했고, `rule_fallback_count: 0`이다
(`outputs/package_02_hybrid/run_report.json`).

**package_02에는 정답지가 없으므로 이 결과가 정답률 향상을 의미하지 않는다.**
아래는 관찰 가능한 사실만 서술한다 — 예측 분포가 바뀌었다는 것과, LLM
신뢰도가 페이지별로 크게 갈렸다는 것.

| 라벨 | rule-only | hybrid |
|---|---|---|
| CREDIT_REPORT | 14 | 15 |
| TITLE_REPORT | 11 | 11 |
| URLA_1003 | 10 | 10 |
| INCOME_DOC | 5 | 5 |
| OTHER | 4 | 3 |

위임 4건의 LLM 판정 (`outputs/package_02_hybrid/page_results.csv`):

| 페이지 | 정규화 텍스트 길이 | LLM 판정 | LLM 신뢰도 |
|---|---|---|---|
| p11 | 0 | OTHER | 0.10 |
| p26 | 0 | OTHER | 0.10 |
| p40 | 0 | OTHER | 0.10 |
| p32 | 671 | CREDIT_REPORT | 0.95 |

텍스트 레이어가 없는 3페이지(p11, p26, p40)는 LLM도 신뢰도 0.10으로
`OTHER`를 반환했다 — 룰이든 LLM이든 텍스트가 아예 없으면 판정 근거가 없다는
뜻이다(12절). 반면 텍스트가 있는 p32는 신뢰도 0.95로 `CREDIT_REPORT`를
반환했다. CREDIT_REPORT 예측 수가 14 → 15로 바뀐 것은 p32 판정 결과이며,
정답지가 없어 이 판정이 실제로 맞는지는 확인할 수 없다.

#### 발견한 문제와 조치: 서식 계열 커버리지 갭

package_01에만 맞춰진 시그니처의 한계가 드러났다. 최초 실행(rules 1.0.0)에서
TITLE_REPORT 예측 수는 4페이지였고, 44페이지 중 18페이지(41%)가 LLM 위임
대상으로 올라갔다. package_01의 위임률은 5%(2/39)였다. package_02에는
정답지가 없으므로 이 4페이지 중 몇 페이지가 실제로 맞았는지는 알 수 없다 —
위임률이 package_01 대비 비정상적으로 높다는 점과 strong 시그니처 매칭이
0건이라는 사실을 근거로 서식 커버리지 갭을 판단했다.

원인은 서식 계열 차이였다. package_01의 표제보험 서류는 CLTA 계열(캘리포니아,
Fidelity National Title)이고 package_02는 ALTA 계열(버지니아, First American
Title)이다. 기존 strong 앵커 6개는 모두 CLTA 전용 문구였고, ALTA 커미트먼트
10페이지 전체에서 strong 매칭이 0건이었다. INCOME_DOC도 같은 구조로,
IRS Wage & Income Transcript 계열에 대응하는 문구가 없었다.

조치로 rules.yaml에 ALTA 계열 9개, IRS 계열 7개 시그니처를 추가했다
(rules_version 1.1.0). 추가한 16개 문자열이 package_01의 추출 텍스트에서
매칭되지 않음을 적용 전에 사전 확인했다. 적용 후 package_01을 다시 실행한
결과 accuracy 0.9744(38/39), macro_f1_supported 0.9853, LLM 위임 대상 2건,
orphan 9건이 기존과 동일하게 유지됐다. 이는 package_01에서 관찰된 회귀가
없다는 의미이며, 다른 문서 계열에 대한 일반화 성능을 보장하지 않는다.

결과적으로 TITLE_REPORT 예측 수는 4 → 11, INCOME_DOC 예측 수는 2 → 5로
변경됐고, 위임 대상은 18 → 4로 줄었다. 수동 서식 검토에서 ALTA/IRS 계열의
사전 커버리지 부족이 확인됐다. 정답지가 없으므로 이 변화가 정답률 향상을
의미하지는 않는다 — 예측 라벨 분포가 바뀌었다는 사실만 확인 가능하다.

#### 발견한 문제: 중복 수록 문서의 인스턴스 분리 실패

package_02에는 동일한 5페이지 ALTA 커미트먼트가 2부 수록되어 있다. 재조립
결과 RD001의 관측 페이지는 `1;1;2;2;3;3;4;4;5;5` 로, 모든 논리 페이지가
2개씩 나타난다. INCOME_DOC(RD003)도 `1;1;2;2` 로 동일한 패턴이며, 실제로는
IRS 트랜스크립트 2건(2024년, 2025년)이다.

`duplicate_policy: keep_both_and_flag` 정책에 따라 중복 페이지가 버려지지 않고
보존된 점은 의도대로 동작한 것이다. package_01에는 중복 문서가 없어 이 정책은
검증되지 않은 경로였고, package_02에서 처음으로 실제 데이터를 만났다.

다만 **2부를 별개 인스턴스로 분리하지는 못했다.** ALTA 서식에는 CLTA의
`PRELIM NO.` 에 해당하는 문서 고유 식별자가 없어 `instance_id` 가 비어 있고,
재조립이 2티어 `form_family_and_marker` 로 떨어진다. 이 티어는 doc_type과
마커만 사용하므로 같은 유형의 서로 다른 문서를 구분할 근거가 없다.
분리하려면 문서 고유 식별자 추출(예: 커미트먼트 파일 번호, 발행일)이
선행되어야 하며, 정답지 없이 검증할 수 없어 이번 범위에서는 구현하지 않았다.
`observed_logical_pages` 에 중복이 그대로 노출되므로 후속 공정에서 감지는 가능하다.

#### 남은 한계

- **텍스트 레이어가 없는 3페이지(p11, p26, p40)**: ALTA Commitment Conditions
  2·3·4 of 4 페이지로, 정규화 후 문자 수가 0이다. `package_01`은 39페이지
  전부 텍스트 레이어가 있었지만, 이 결과를 "전체 데이터셋에서 OCR이
  불필요하다"로 일반화할 수 없다는 것이 package_02에서 드러났다. 텍스트
  기반 룰로는 원리상 분류할 수 없어 SHORT_TEXT 조건으로 LLM에 위임됐지만,
  애초에 보낼 텍스트가 없으므로 텍스트 전용 LLM을 연결해도 이 3페이지는
  해결되지 않는다(API 키도 없어 실제 호출은 발생하지 않았다). OCR 또는
  Vision 기반 처리가 필요하지만 이번 제출 범위에서는 구현하지 않았다.
- **분류 경계가 모호한 1페이지(p32)**: Experian Employment Verification 보고서로,
  신용조회기관 발행물이지만 신용보고서 본체는 아니다. 5개 라벨 정의만으로는
  단정할 수 없어 위임 대상으로 남겼다.
- **orphan 11페이지**: 현재 확보한 페이지 마커와 식별자만으로 특정
  reconstructed document에 귀속하지 못한 상태다. 잘못된 문서에 강제로 결합하지
  않고 미해결 상태로 노출했다. orphan에는 실제로 독립된 페이지, 마커가 없는
  부속 페이지, 텍스트 레이어가 없어 구조 정보를 추출하지 못한 페이지가 섞일 수
  있으므로 모두 정상 페이지라고 단정하지 않는다.
- **역할 분리**: 텍스트는 있으나 룰 근거가 약한 페이지는 텍스트 LLM 위임
  후보다. 반면 package_02의 p11·p26·p40처럼 텍스트 레이어가 없거나, package_01의
  p8처럼 내용이 제거되어 판정 근거가 거의 남지 않은 페이지는 텍스트 LLM만으로
  해결하기 어렵고 OCR 또는 Vision 모델이 필요하다. 분류는 됐지만 문서
  인스턴스 식별자가 없는 페이지는 LLM 분류 문제가 아니라 재조립 식별자 문제라,
  문서 고유 식별자 또는 좌표 기반 필드 추출이 필요하다.
- 시그니처 추가는 **발견된 서식 계열 2종에 대한 대응**이다. 세 번째 계열을
  만나면 같은 문제가 재발할 수 있다. 이 경우에도 해결 방향은 위 세 경우를
  나누어 판단해야 한다.

## 7. 그룹핑

과제의 필수 산출물은 **물리적 연속 그룹핑**이다. 인접한 물리 페이지가 같은
`doc_type`으로 예측되면 하나의 그룹으로 묶는다.

**실측 결과: 39개 그룹, 전부 `page_count=1`.** 이는 알고리즘의 실패가 아니라
`package_01`이 scatter 방식으로 셔플되어 인접한 두 페이지가 같은 유형인
경우가 38쌍 중 0쌍이기 때문이다. 병합하거나 보정하지 않고 실제 구조 그대로
출력한다.

**논리 문서 재조립**은 확장 기능이다. `doc_type`, `instance_id`, `marker_style`,
`marker_total` 네 요소를 조합한 키로, 식별자와 마커가 모두 있으면 확정 결합
(tier 1), 식별자 없이 마커만 일치하면 결합하되 `AMBIGUOUS_ATTACHMENT`로 표시
(tier 2), 증거가 부족하면 결합하지 않고 orphan으로 남긴다(tier 3). 네 요소가
모두 필요한 이유: URLA_1003과 CREDIT_REPORT는 `package_01`에서 marker_total이
모두 11이므로, marker_style이 이 둘을 실제로 구분하는 요소가 된다.

실측 결과 3개 문서로 재조립됐다:

```
doc_id  doc_type       instance_id   marker_style  expected  observed  completeness    issues
RD001   CREDIT_REPORT  -             PAGE_N_OF_M   11        11        COMPLETE        AMBIGUOUS_ATTACHMENT
RD002   URLA_1003      URLA_1003#1   N_OF_M        11        11        COMPLETE        -
RD003   TITLE_REPORT   -             CLTA_PAGE_N   -          8        UNKNOWN_EXTENT  AMBIGUOUS_ATTACHMENT
```

URLA_1003만 `Freddie Mac Form 65` 식별자가 추출되어 tier 1로 확정됐다. 나머지
두 유형은 식별자가 전혀 추출되지 않아(원인은 아래 참고) 마커만으로 결합했다.
증거가 부족한 9페이지는 orphan으로 미해결 상태로 노출했다 — 강제로 붙이지 않는다.

논리→물리 페이지 매핑(일부 발췌, 전체는 `reconstructed_documents.csv`):

```
RD001 CREDIT_REPORT: 1->9, 2->31, 3->39, 4->5, 5->33, 6->19, 7->1, 8->15, 9->27, 10->11, 11->7
RD002 URLA_1003:     1->6, 2->38, 3->10, 4->20, 5->4, 6->14, 7->30, 8->24, 9->35, 10->2, 11->26
RD003 TITLE_REPORT:  1->32, 2->34, 3->37, 4->12, 5->22, 6->16, 7->18, 8->28
```

세 문서 모두 `is_contiguous=N`이다 — 논리 페이지가 물리적으로 완전히 흩어져
있다는 뜻이며, 이것이 물리적 연속 그룹핑이 39개 단일 그룹을 내는 이유이기도
하다. `RD001`(9>7)과 `RD003`(32>28)처럼 문서의 첫 논리 페이지가 마지막 논리
페이지보다 물리적으로 뒤에 오는 경우가 있는데, scatter-shuffled 입력에서 관찰될 수
있는 값이다. CLI 출력에는 `physical_span`, `is_contiguous` 컬럼과 안내문을 함께
표시해 오해를 막는다.

`loan_number`/`report_id`/`prelim_number`는 39페이지 어디서도 추출되지 않았다
(`form_family`만 URLA 11페이지에서 추출됨). 레이아웃 기반 PDF에서 라벨과 값의
텍스트 스트림상 위치가 떨어져 있어 정규식이 매칭되지 않기 때문이다 — 자세한
원인과 미수정 이유는 12절 7항 참고.

## 8. 오답 분석

`package_01`에서 rule-only 오분류는 1건, hybrid 오분류도 1건이지만 서로
다른 페이지다(6절).

- **p8 — rule-only의 유일한 오분류, hybrid에서는 LLM이 정정** (정답
  `TITLE_REPORT`). 지도 이미지가 익명화 과정에서 텍스트만 남아 시그니처와
  마커가 모두 없다(`matched_class_count=0`, `rule_score=0.00`) — 룰은
  `OTHER`로 예측했다. 실제 LLM 호출에서는 신뢰도 0.95로 `TITLE_REPORT`를
  정확히 판정했다(`outputs/package_01_hybrid/page_results.csv`). 남은
  텍스트만으로도 LLM에게는 판정 근거가 있었다는 뜻이며, "룰과 LLM 모두 판정
  근거가 없다"고 봤던 이전 서술은 `call_llm`이 미구현이었을 때의 가정이었을
  뿐 실측과 다르다.
- **p36 — rule-only에서는 라벨이 정답과 일치했으나 hybrid에서는 LLM이 오판**
  (정답 `INCOME_DOC`). 손익계산서 제목이 이미지로 렌더링되어 텍스트
  레이어에 없고, 매칭된 룰 신호가 weak 시그니처(`CTEC`) 하나뿐이라
  (`rule_score=0.25`) `LOW_RULE_SCORE`로 위임됐다. rule-only 단계에서는 룰이
  이 페이지를 맞혔지만, 실제 LLM 호출은 신뢰도 0.70으로 `OTHER`를 반환해
  hybrid의 새 오분류가 됐다. 위임이 항상 정답률 개선으로 이어지지는 않는다는
  실측 사례다.
- **CREDIT_REPORT 부속 페이지 7장 orphan** — 분류 자체는
  `Credit Score Disclosure`, `XACTUS` 등 시그니처로 정확했지만 페이지 마커가
  없어 논리 재조립에서 orphan으로 미해결 상태로 노출했다. 분류 실패가 아니라
  재조립 정보 부족이다.

p8과 p36을 함께 보면, LLM 위임은 룰의 오분류를 정정할 수도 있고(p8) 룰의
정답을 뒤집을 수도 있다(p36). package_01만으로는 위임 2건, package_02까지
합쳐도 6건뿐이라 이 트레이드오프의 방향성을 일반화할 수 없다(12절).

rule-only 기준 세 사례를 포함한 상세 분석은
[`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) 참고 —
`docs/ERROR_ANALYSIS.md`는 rule-only와 hybrid 두 모드의 결과를 모두 반영한다.

## 9. AI 사용

**런타임 분류**: `llm_classifier.py`는 설정 로딩, PII 마스킹, 프롬프트 구성,
배치 분할, 응답 검증, 재시도/fallback 오케스트레이션에 더해 **실제 OpenAI
Python SDK 네트워크 호출부(`call_llm`)까지 구현했다.**
`client.chat.completions.parse`에 `response_format=LLMBatchResponse`(pydantic
모델)를 그대로 전달해 Structured Outputs로 스키마를 강제하고, SDK가 반환하는
`message.parsed`(이미 검증된 `LLMBatchResponse`)를 그대로 사용한다.
temperature는 0으로 고정하고 `model`/`timeout_seconds`는 `LLMRuntimeConfig`에서
가져온다. mock 기반 테스트(`call_llm`을 monkeypatch)는 이전과 동일하게
유지된다 — 실제 네트워크 호출은 테스트에 포함되지 않는다.

- **프로바이더**: OpenAI Python SDK(`openai>=1.40`)를 그대로 쓰되, 환경변수
  `LLM_BASE_URL`로 엔드포인트를 교체할 수 있다. 비어 있으면 실제 OpenAI
  엔드포인트(`api.openai.com`)를 사용한다. `LLMRuntimeConfig`에는
  `api_key`/`base_url` 필드를 추가하지 않았다 — 둘 다 `call_llm` 안에서
  `load_llm_config`와 같은 방식으로 환경변수(`LLM_API_KEY`, `LLM_BASE_URL`)를
  직접 읽어 그 호출 한 번에만 쓰고 저장하지 않는다. **제출 시점 실행에는
  Google Gemini의 OpenAI 호환 엔드포인트**
  (`https://generativelanguage.googleapis.com/v1beta/openai/`)와 **모델
  `gemini-2.5-flash`**를 사용했다 — `LLM_API_KEY`가 Gemini AI Studio에서
  발급받은 키이기 때문이다. 실제 OpenAI 엔드포인트로는 호출을 검증하지
  않았다.
- **API 키 없음**: `LLMRuntimeConfig.enabled=False`가 되어 `hybrid` 모드가
  `LLM_DISABLED` 경고와 함께 rule-only와 동일하게 자동 강등된다(기존과 동일한
  동작). 예외 없이 완주하며 `run_report.json`의 `mode`는 `hybrid`로 유지된다.
- **API 키 있음**: 실제 네트워크 호출을 수행한다. `package_01`은 위임 대상
  2페이지(p8, p36), `package_02`는 위임 대상 4페이지(p11, p26, p32, p40) 전량
  실제로 호출했고, 두 패키지 모두 `rule_fallback_count: 0`으로 전량
  성공했다(`outputs/package_01_hybrid/run_report.json`,
  `outputs/package_02_hybrid/run_report.json`). 호출이 실패하면(인증 실패,
  타임아웃, 스키마 불일치 등) `resolve_delegated_pages`가 재시도
  (`max_retries`)를 소진한 뒤 `decision_source=RULE_FALLBACK`,
  `warnings: ["LLM_CALL_FAILED"]`로 처리하고 최종 라벨은 룰 결과를 유지한다 —
  이 경로는 기존 mock 테스트로 검증돼 있고 실제 호출 구현 후에도 로직은
  그대로다.
- **테스트된 범위**: PII 마스킹, 요청/프롬프트 구성, 배치 분할, 응답 검증(잘못된
  라벨 처리, 응답 개수 불일치 포함), 재시도, rule fallback — 전부 `call_llm`을
  monkeypatch한 mock 기반 pytest로 검증한다.
- **테스트되지 않은/제한적으로만 확인된 범위**: 실제 OpenAI 엔드포인트로의
  호출(Gemini 호환 레이어만 사용), 대규모 배치·비용·지연시간·응답 안정성(두
  패키지 합쳐 6페이지, 배치 2회만 확인), `gemini-2.5-flash` 외 다른 모델의
  동작.

`outputs/package_01_hybrid`와 `outputs/package_02_hybrid`는 **실제 LLM 추론
결과**다 — `llm_config.enabled: true`이고 각각 `llm_called: 2`, `llm_called: 4`,
두 패키지 모두 `rule_fallback_count: 0`으로 확인된다. (이전에는 API 키 없이
자동 강등만 확인한 산출물이었으나, `call_llm` 구현 후 재실행해 교체했다.)

**개발 보조**: 이 코드베이스는 Claude Code와 Codex를 사용해 작성했다.
`src/schema.py`와 `config/rules.yaml`을 인터페이스 계약으로 고정하고
`.claude/settings.json`의 에이전트 권한 정책으로 계약 파일 수정과 Git 조작을
차단했다. 모든 함수 출력은 로컬 실행과 결정론적 pytest로 검증했다. AI가 전체를
자동으로 작성하거나 검증을 완결한 것은 아니며, 각 모듈은 실제 실행 결과를
대조해 확인했다.

## 10. 보안 및 데이터 정책

- 원본 PDF는 저장소에 포함하지 않는다(`.gitignore`).
- `ExtractedPage`만 페이지 원문과 원본 식별자를 보유하며 프로세스를 벗어나지
  않는다. 그 이하 모든 산출 모델(`PageResult` 등)은 비식별이다.
- `instance_id`는 실행 중 부여한 대체 ID(`URLA_1003#1` 형식)이며 원본 대출번호를
  저장하지 않는다.
- `RunReport.source_pdf_name`은 파일명(basename)만 저장한다.
- 저장 직전 `check_no_pii` 헬퍼가 SSN·이메일 패턴을 검사하고, 발견되면 저장을
  중단한다. 통과하는 것이 정상 경로이며, 걸린다면 상위 로직의 버그를 뜻한다.
- 시각화(`visualize.py`)는 라벨별 색상 타일만 사용한다. 실제 PDF를 렌더링하거나
  썸네일을 만들지 않는다 — 이미지에 원문이 노출될 경로 자체를 차단한다.
- LLM 요청/응답 원문은 로그와 출력 파일 어디에도 저장하지 않는다.
  `LLMCallRecord`는 페이지 번호, 시도 횟수, 성공 여부, 경고 코드만 기록한다.
- `.claude/settings.json`이 `git add`/`commit`/`push`, `rm`,
  `schema.py`/`rules.yaml` 수정, `.env` 읽기를 에이전트 권한 정책으로 차단한다.

## 11. 선택하지 않은 방식

주요 항목만 요약한다.

- **전 페이지 OCR** — 39페이지 전부 텍스트 레이어가 있어(최소 68자) 불필요했다.
- **TF-IDF + 전통 ML** — `INCOME_DOC` 학습 표본 1개, `OTHER` 0개로 과적합이
  발생할 가능성이 크다.
- **전 페이지 LLM 위임** — 재현성 저하와 비용·속도 문제로 제외했다. package_01에서
  37/39페이지는 위임 불필요로 판단됐다.
- **벤더 주소를 CREDIT_REPORT 시그니처로 사용** — 데이터셋 과적합이라 제외했다.
  주소 없이도 18페이지 전체가 커버된다.

전체 비교표와 판단 근거는 [`docs/APPROACH.md`](docs/APPROACH.md) 참고.

## 12. 한계 및 개선 방향

1. `package_01`을 룰 개발과 자체 평가에 함께 사용했다. 0.9744는 개발 데이터
   기준이며 독립 일반화 성능을 보장하지 않는다. `package_02`도 rules 1.1.0
   수정에 사용됐으므로 독립 holdout이 아니며, 정답지가 없어 accuracy는 판단할
   수 없다.
2. `package_01`의 클래스 분포가 작고 불균형하다. `INCOME_DOC` 정답은 1페이지뿐이고
   `OTHER` 정답은 0페이지라 해당 클래스들의 지표와 분류 경로 해석에 한계가 있다.
3. 실제 LLM 네트워크 호출부(`call_llm`)는 구현되어 있고 두 패키지 모두에서
   실행을 확인했지만(9절), 검증 규모가 작다 — 위임 대상은 package_01 2건,
   package_02 4건, 총 6페이지·배치 2회뿐이다. 대규모 배치 처리, 재시도가
   실제로 발동하는 경우, 응답 지연시간·비용, 응답 안정성(같은 입력을 반복
   호출했을 때 동일한 결과가 나오는지)은 측정하지 않았다. 또한 실제 OpenAI
   엔드포인트가 아니라 Gemini의 OpenAI 호환 레이어로만 확인했으므로,
   `gpt-5-mini`를 포함한 실제 OpenAI 모델의 동작은 여전히 검증되지 않았다.
4. OCR fallback이 없다. `package_01` 39페이지는 전부 텍스트 레이어가 있었지만,
   `package_02`의 p11/p26/p40은 정규화 텍스트 길이가 0이다(6절). 따라서
   "이 데이터셋은 OCR이 불필요하다"는 결론을 전체 패키지로 일반화할 수 없다.
   텍스트가 없는 페이지는 텍스트 전용 LLM을 연결해도 해결되지 않는다 — 실제
   LLM 호출로도 이를 확인했다: p11/p26/p40 세 페이지 모두 신뢰도 0.10으로
   `OTHER`를 반환했다(`outputs/package_02_hybrid/page_results.csv`, 6절).
   OCR 또는 Vision 기반 처리가 필요하지만 이번 제출 범위에서는 구현하지
   않았다.
5. 다중 문서 인스턴스 재조립 실패가 `package_02`에서 실제로 확인됐다(6절).
   동일한 마커 구조를 가진 ALTA 커미트먼트 2부와 IRS 트랜스크립트 2건이 각각
   하나의 reconstructed document로 병합됐다. `keep_both_and_flag` 정책 덕분에
   중복 페이지 자체는 삭제되지 않고 보존됐지만, 같은 유형·같은 마커 구조를
   가진 두 인스턴스를 분리하지는 못했다. 개선 방향은 문서 고유 식별자
   추출이며, 이번 범위에서는 구현하지 않았다.
6. `marker_affinity`는 현재 분류 점수 계산에 사용되지 않는다. `src/rule_classifier.py`의
   `classify_page`는 strong/weak 시그니처만 점수화하고, 마커는 재조립 구조
   신호로만 보존한다.
7. 별도 `--rules` 파일에 대한 `never_strong` 검증은 기본 import 시 검증과 동일하게
   자동 보장되지 않는다. `src.rule_classifier` import 시 기본 `config/rules.yaml`은
   검증하지만, CLI의 `--rules` 경로는 `load_rules(args.rules)`로 읽어 그대로
   pipeline에 전달된다.
8. 문서 식별자(`loan_number` 등) 추출이 실패한다. 레이아웃 기반 PDF에서 라벨과
   값의 텍스트 스트림상 근접성이 보장되지 않기 때문이다. 개선 방향은 좌표 기반
   추출(`get_text("words")`)이다.
9. p8처럼 내용이 제거되어 문맥이 거의 남지 않은 페이지는 룰로는 처리하기
   어렵다 — 다만 p8은 실제 LLM 호출로는 신뢰도 0.95로 정확히 분류돼(8절),
   남은 텍스트가 극히 적어도 텍스트 LLM이 항상 무력하지는 않다는 것이 실측으로
   확인됐다. 반대로 텍스트 자체가 없는 페이지(4항의 p11/p26/p40)는 텍스트
   LLM으로도 해결되지 않는다 — 문맥이 "거의 없는" 경우와 "전혀 없는" 경우를
   구분해야 한다.
10. 인명 마스킹은 정규식 휴리스틱이라 완전성을 보장하지 못한다. 또한 소득
    서류처럼 금액 구조 자체가 분류 신호인 문서(p36)에서는 무차별 숫자 마스킹이
    그 신호를 파괴한다는 점도 확인했다.
11. `NARROW_RULE_MARGIN` 조건은 실데이터에서 발동하지 않았고 synthetic 테스트에서만
    동작을 확인했다.
12. `requirements.txt`는 `>=` 최소 버전 위주라 완전한 의존성 재현성을 보장하지
    않는다.
13. 회전 무관 텍스트 추출은 현재 데이터셋에서만 확인된 사실이며 모든 PDF에
    일반화할 수 없다.
14. CI가 구성되어 있지 않다. 원본 PDF를 저장소에 포함하지 않으므로(10절) CI에서
    `package_01` 평가를 수행할 수 없고, 지금까지 모든 테스트는 로컬에서만
    실행했다(102 passed, 1 warning). 개선 방향은 PDF 없이도 동작하는 synthetic
    fixture 기반 결정론적 테스트만 CI로 분리하는 것이다.
15. LLM 위임이 항상 정답률 개선으로 이어지지는 않는다. `package_01`의 p36에서
    실제 LLM 호출이 룰보다 나쁜 판정(`INCOME_DOC` → `OTHER`, 신뢰도 0.70)을
    내려 hybrid의 새로운 오분류가 됐다(6·8절). 위임 대상이 6페이지뿐이라 이
    트레이드오프의 방향성을 일반화할 수 없다.

실 운영에서는 필드 단위 선택적 마스킹이나 온프레미스 모델이 필요하다.
