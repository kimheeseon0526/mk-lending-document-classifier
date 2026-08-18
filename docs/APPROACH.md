# Approach

주요 기술적 의사결정과 검토했으나 채택하지 않은 대안을 기록한다. `README.md`의
아키텍처·평가 절 요약을 전제로, 여기서는 각 결정의 배경·대안·근거·한계를
실측값과 함께 상세히 다룬다. 검증 근거는 `outputs/package_01/evaluation.json`,
`outputs/package_01/run_report.json`, `outputs/package_01_hybrid/run_report.json`,
`outputs/package_02/run_report.json`, `outputs/package_02_hybrid/page_results.csv`,
`docs/ERROR_ANALYSIS.md`에 있다. `docs/ERROR_ANALYSIS.md`는 hybrid 실행 결과
반영 전(rule-only 기준) 문서다.

아래 결정 대부분은 `package_01` 데이터만으로 내려졌다. `package_02`는
정답지가 없어 독립 검증셋이 아니라 룰 적응(서식 커버리지 확장)에 사용한
추가 입력이며(README 6절), `package_02`에서 새로 확인된 한계는 각 결정의
Limitation에 별도로 표기했다.

## Decision: 룰 우선 + 저신뢰 페이지만 LLM 위임

### Context
페이지마다 5개 라벨 중 하나를 예측해야 한다. 키워드 시그니처 기반 판정으로는
표준 시그니처가 텍스트 레이어에 없는 일부 페이지(제목이 이미지로 렌더링된
경우, 지도 이미지만 남은 경우)를 판정할 수 없다.

### Decision
`rule_classifier.classify_page`로 전 페이지를 1차 채점하고, `should_delegate`가
True인 페이지만 LLM에 위임하는 하이브리드 구조를 채택했다.

### Alternatives
- 전 페이지 LLM 위임
- TF-IDF + 전통 ML 분류기
- 전 페이지 Vision LLM (페이지 이미지 자체를 모델에 전달)

### Reason
`package_01` 39페이지 중 37페이지가 룰 시그니처만으로 결정론적으로 확정된다
(단독 확정 정답 37/39, 충돌 0, 오답 0). 위임 대상은 2장(p8, p36)뿐이다. 전
페이지 LLM 위임은 실행마다 결과가 달라질 수 있어 재현성이 떨어지고 비용·속도
부담이 있다. TF-IDF/전통 ML은 `INCOME_DOC` 학습 표본이 1개, `OTHER`가 0개라
과적합이 확정적이다.

### Limitation
위임된 2장의 실제 모델 판정이 항상 옳지는 않다 — p8은 LLM이 정정했지만
(`TITLE_REPORT`, 신뢰도 0.95) p36은 LLM이 룰의 정답을 뒤집었다(`OTHER`, 신뢰도
0.70; README 6·8절). 또한 이 위임 경로는 텍스트 기반이므로 정규화 텍스트
길이가 0인 페이지에는 원리상 통하지 않는다 — `package_02`의 p11/p26/p40이
이 경우로, 실제 LLM 호출도 신뢰도 0.10으로 `OTHER`를 반환해 텍스트 전용
LLM으로는 해결되지 않는다는 예상이 실측으로 확인됐다(README 6·12절).
OCR/Vision 기반 처리가 필요하지만 구현하지 않았다.

---

## Decision: 물리적 연속 그룹핑과 논리 재조립을 두 축으로 분리

### Context
과제는 물리적 연속 그룹핑을 필수 산출물로 요구한다. `package_01`은 PDF
메타데이터상 `strategy=scatter, seed=42`로 셔플됐고, 인접한 두 페이지가 같은
유형인 경우가 38쌍 중 0쌍이다. 이 조건에서 물리적 인접성만으로는 원본 문서를
하나도 복원할 수 없다.

### Decision
`build_physical_groups`(물리적 연속 그룹핑, 필수)와 `reconstruct_documents`
(논리 문서 재조립, 확장)를 서로 대체하지 않는 독립된 산출물로 분리했다.

### Alternatives
- 물리적 인접성만으로 원본 복원 시도

### Reason
인접 동일 유형 0/38쌍이라는 실측 특성 때문에 물리적 그룹핑만으로는 39개의
1페이지 그룹만 나온다. 마커(`marker_page`)와 식별자(`instance_id`)를 근거로
한 별도의 논리 재조립 단계를 두어 3개 문서(`RD001`~`RD003`)를 복원했다.

### Limitation
논리 재조립은 과제의 필수 요구사항이 아니라 확장 기능이며, 물리적 그룹핑
결과(39개 그룹)를 대체하지 않는다. 두 산출물은 각각 `physical_groups.csv`,
`reconstructed_documents.csv`로 항상 함께 출력된다.

---

## Decision: GroupingKey를 네 요소 조합으로 정의

### Context
`marker_total` 단독으로 문서를 그룹핑하면 서로 다른 문서가 병합될 위험이
있다.

### Decision
`GroupingKey = (doc_type, instance_id, marker_style, marker_total)` 네
요소를 모두 조합한 복합 키로 문서를 그룹핑한다.

### Alternatives
- `marker_total` 단독
- `doc_type` 단독
- `instance_id` 단독

### Reason
`URLA_1003`과 `CREDIT_REPORT`는 `package_01`에서 `marker_total`이 모두
11이라 그 값만으로는 두 문서가 병합된다. `instance_id` 단독도 마찬가지로
병합되는데, 같은 대출 건의 loan number(`990145627`)가 `CREDIT_REPORT` 14장과
`URLA_1003` 4장에서 공유되기 때문이다. `marker_style`이 "Page N of M"
(`CREDIT_REPORT`)과 "N of M"(`URLA_1003`)을 실제로 구분하는 요소다.

### Limitation
같은 4요소 조합을 가진 서로 다른 문서 인스턴스가 2개 이상 존재하는 경우,
`package_01`은 유형별 문서가 1개씩이라 이 시나리오 자체가 없어 원래
검증되지 않았다. `package_02`에서 실제로 이 경우가 발생했다 — 동일한 마커
구조를 가진 ALTA 커미트먼트 2부와 IRS 트랜스크립트 2건이 각각 하나의
reconstructed document로 병합됐다(package_02에서 확인됨, README 6절).
`keep_both_and_flag` 정책 덕분에 중복 페이지 자체는 삭제되지 않고
보존됐지만, 현재 `GroupingKey`로는 두 인스턴스를 분리할 근거가 없다는
한계가 실측으로 확인됐다. 분리하려면 문서 고유 식별자 추출이 선행돼야
한다.

---

## Decision: CREDIT_REPORT 시그니처로 TUC/EXP/EQX 채택, 벤더 주소는 제외

### Context
`CREDIT_REPORT` 18페이지를 표준 시그니처(`XACTUS`, `Credit ReportX`)로
검사하면 단독 확정 정답 28/39, 미검출 11장(그중 9장이 `CREDIT_REPORT`)이
나온다. 원인은 벤더 로고가 래스터 이미지로 삽입되어 텍스트 레이어에 없기
때문이다.

### Decision
신용조회기관 코드 `TUC/EXP/EQX`를 strong 시그니처로 채택했다.

### Alternatives
- `XACTUS`/`Credit ReportX` 문자열 유지 (텍스트 레이어에 없어 미검출 발생)
- 벤더 주소(`370 Reed Rd., Suite 100 Broomall`)를 시그니처로 사용

### Reason
`TUC/EXP/EQX` 채택 후 재측정 결과 단독 확정 정답 37/39, 미검출 2(p8, p36),
충돌 0, 오답 0으로 개선됐다. 벤더 주소도 동일 페이지를 커버하지만 이 특정
벤더의 주소 문자열에 의존하는 것은 데이터셋 과적합이며, 주소 없이도
`TUC/EXP/EQX`만으로 18페이지 전체가 커버되므로 채택하지 않았다.

### Limitation
`XACTUS` 로고가 텍스트 레이어에 없는 것이 이 익명화 샘플의 특성인지, 원본
벤더 PDF 전반의 특성인지는 확인되지 않았다.

---

## Decision: PII 마스킹에서 금액 구조 보존

### Context
`INCOME_DOC`(p36)은 표준 소득 서류 키워드(Paystub, W-2, 1040, 1099 등)가
전혀 없고, 매칭된 룰 신호는 weak 시그니처 `CTEC` 하나(`rule_score=0.25`)뿐이라
LLM 위임 대상이다.

### Decision
`never_mask_regex`로 금액 표현(`$231,239.00`, `49,720.00` 형식)을 마스킹
패턴이 덮어쓸 수 없는 보호 구간으로 지정해 항상 보존한다.

### Alternatives
- 8자리 이상 숫자를 포함한 모든 숫자를 무차별로 `[NUM]` 마스킹

### Reason
무차별 숫자 마스킹을 p36에 시뮬레이션하면 `$[NUM]`, `CTEC #A[NUM]`처럼 모든
금액이 사라진다. p36의 유일한 분류 신호는 매출·비용 항목·순이익으로 이어지는
금액의 구조적 관계 자체이므로, 숫자를 마스킹하면 LLM이 손익계산서로 판정할
근거가 사라진다. 즉 무차별 마스킹은 LLM 위임이 필요한 바로 그 페이지의 신호를
파괴한다.

### Limitation
현재 정책은 이 데이터셋에서 관찰된 사실(금액 구조가 분류 신호)에 맞춘
것이다. 실 운영에서는 필드 단위 선택적 마스킹이나 온프레미스 모델이 필요하다.

---

## Decision: OTHER의 support=0을 N/A로 처리하고 macro 평균에서 제외

### Context
`package_01`에는 `OTHER` 정답이 0개다. `OTHER`의 recall/F1은 수학적으로
정의되지 않는다(0/0).

### Decision
support=0인 클래스의 recall/F1을 `0.0`이 아니라 `None`(N/A)으로 두고,
`macro_f1_supported` 계산에서 제외한다. precision은 계산 가능하므로(예측이
있으면) 그대로 산출한다.

### Alternatives
- 정의되지 않은 지표를 `0.0`으로 채워 계산

### Reason
`0.0`으로 채우면 macro 평균이 `(1.0+1.0+1.0+0.9412+0.0)/5 = 0.788`로
왜곡된다. N/A 처리 시 `0.9853`을 얻는다. `OTHER`로 잘못 예측한 1건(p8)은
false positive로 집계되며 precision은 `0.0`으로 정상 산출된다 — recall/F1만
정의 불가능한 것이지 precision까지 무의미한 것은 아니다.

### Limitation
`OTHER` 분류 경로 자체가 정답 데이터 부재로 전혀 검증되지 않았다.

---

## Decision: matched_class_count >= 2 조건으로 narrow margin 검사 게이팅

### Context
`rule_margin` 값 하나만으로는 "실제로 두 유형이 경합하는 상황"과 "매칭된
시그니처가 아예 없는 상황"을 구분할 수 없다. 둘 다 낮은 margin 값을 낼 수
있다.

### Decision
`rule_margin`이 임계값 미만이어도, 매칭된 클래스 수(`matched_class_count`)가
2 이상일 때만 `NARROW_RULE_MARGIN`을 위임 사유로 사용한다.

### Alternatives
- 게이팅 없이 `rule_margin` 임계값만으로 판단

### Reason
`matched_class_count`가 0인 페이지(예: p8)는 margin이 `0.00`이지만 이는
"경합"이 아니라 "증거 부재"다. 증거 부재는 `SHORT_TEXT`/`LOW_RULE_SCORE`가
이미 담당하므로 `NARROW_RULE_MARGIN`까지 붙이면 위임 사유가 상황을 잘못
설명하게 된다. 게이팅 적용 후 `package_01` 실측: `matched_class_count>=2`인
페이지는 p4, p11 두 장뿐이며 margin이 각각 0.86, 0.80으로 임계값(0.20)을
크게 상회해 `NARROW_RULE_MARGIN`이 발동하지 않았다.

### Limitation
이 조건은 `package_01` 실데이터에서 한 번도 발동하지 않았다 — synthetic
conflict 테스트로만 동작을 확인했다. "경합 사례 자체가 없었다"는 뜻은 아니다.
p4, p11처럼 서로 다른 유형의 시그니처가 동시에 매칭된 사례는 존재했지만, 한쪽
시그니처가 strong이라 short-circuit 단계에서 이미 확정되어 margin 검사에
도달하지 않았다.

---

## Decision: 문서 식별자 추출 실패를 수정하지 않은 판단

### Context
`loan_number`/`report_id`/`prelim_number`가 39페이지 어디서도 추출되지
않는다(`form_family`만 URLA 11페이지에서 추출됨). 레이아웃 기반 PDF에서 라벨과
값의 공간적 근접성이 텍스트 스트림 순서에서 보장되지 않기 때문이다. 예를 들어
화면상 `Loan Number: 990145627`로 보이지만 추출 텍스트는 `Loan Number:` 뒤에
다른 필드가 먼저 오고 `990145627`은 한참 뒤에 나온다.

### Decision
`Loan\s*Number[:\s]*(\d{6,12})` 등 현재 정규식을 느슨하게 만들지 않고 유지한다.

### Alternatives
- 정규식의 허용 범위를 넓혀 라벨에서 더 멀리 떨어진 숫자도 매칭

### Reason
`package_01`은 doc_type별 문서가 1개씩이므로 `(doc_type, marker_style,
marker_total)` 세 요소만으로 이미 완전히 분리된다. 정규식을 느슨하게 만들면
Client Code, Docket 번호, 계좌번호 등을 오인식할 위험이 있고, 이를 검증할
다중 인스턴스 데이터가 이 프로젝트에는 없다.

### Limitation
같은 유형 문서가 2개 이상 존재하는 실제 상황에서는 이 결정이 재조립 실패로
이어질 수 있다. 개선 방향은 좌표 기반 추출(`get_text("words")`)로 라벨과
값의 화면상 위치를 직접 비교하는 것이다.

---

## Decision: 호출부 없는 골격을 먼저 검증한 뒤 실제 네트워크 호출을 붙임

### Context
하이브리드 파이프라인은 API 키 유무와 무관하게 항상 완주해야 하고, 위임·검증·
재시도·fallback 경로가 먼저 검증돼야 나중에 실제 호출을 붙였을 때 바로 신뢰할
수 있다.

### Decision
1단계로 설정 로딩, PII 마스킹, 프롬프트 구성, 배치 분할, 응답 검증, 재시도,
fallback 오케스트레이션까지 전부 구현·테스트하고 `call_llm`은
`NotImplementedError`로 남겨 골격만 검증했다. 이후 2단계로 `call_llm`을
OpenAI Python SDK의 `chat.completions.parse` + `response_format=LLMBatchResponse`
(Structured Outputs)로 실제 구현했다. `LLMRuntimeConfig`에는 `api_key`/`base_url`
필드를 추가하지 않고(스키마를 인터페이스 계약으로 고정한 원칙 유지, README
9·10절), 두 값 모두 `call_llm` 안에서 `load_llm_config`와 같은 방식으로
환경변수(`LLM_API_KEY`, `LLM_BASE_URL`)를 직접 읽어 그 호출 한 번에만 쓰고
저장하지 않는다.

### Alternatives
- 골격 없이 처음부터 실제 호출부까지 한 번에 구현
- `LLMRuntimeConfig`에 `api_key`/`base_url` 필드 추가

### Reason
골격을 먼저 검증한 이유는 원래 판단과 동일하다 — API 키가 없으면
(`llm enabled: False`) 파이프라인 전체가 rule-only와 동일하게 자동 강등되어
예외 없이 완주해야 하고, 이 경로는 실제 호출 구현 유무와 무관하게 항상
성립해야 한다. `LLMRuntimeConfig`에 필드를 추가하지 않은 이유는 이 스키마가
저장 산출물과 함께 인터페이스 계약으로 고정되어 있고, 자격 증명을 여기 흘려
넣으면 어딘가에서 직렬화·로그될 위험이 생기기 때문이다 — 대신 `call_llm`
호출 시점에만 환경에서 읽고 그 함수 스코프를 벗어나지 않게 했다. 실제
구현 후 두 패키지 모두에서 실행해 `rule_fallback_count: 0`(package_01 2건,
package_02 4건 전량 성공)을 확인했다(README 6·9절).

### Limitation
검증 규모가 작다 — 총 6페이지, 배치 2회뿐이다. 대규모 배치, 실제 재시도가
발동하는 사례, 비용·지연시간, 응답 안정성은 측정하지 않았다. 실제 OpenAI
엔드포인트가 아니라 Gemini의 OpenAI 호환 레이어(`base_url` 교체)로만
검증해, `gpt-5-mini`를 포함한 실제 OpenAI 모델의 동작은 여전히 확인되지
않았다. 또한 위임이 항상 룰보다 나은 것은 아니다 — package_01 p36에서는
LLM이 룰의 정답(`INCOME_DOC`)을 `OTHER`로 뒤집었다(신뢰도 0.70).

---

## Decision: 시각화를 색상 타일로 한정하고 PDF 썸네일을 만들지 않음

### Context
분류 결과를 시각화해야 하는데, 페이지 원문이나 식별자가 이미지 파일에 노출될
경로를 만들지 않아야 한다.

### Decision
실제 PDF 페이지를 렌더링하거나 썸네일을 만들지 않고, `DocType`별 고정 색상
타일만으로 페이지 스트립과 confusion matrix를 그린다.

### Alternatives
- 축소 렌더링한 페이지 썸네일 위에 라벨을 오버레이

### Reason
`LLMPageRequest.excerpt`는 `mask_text`를 거쳐 PII를 제거하지만, 이 프로젝트에는
이미지 렌더링 결과물을 마스킹하는 별도 파이프라인이 없다. 썸네일 방식을
택하면 그 파이프라인이 새로 필요해진다. 색상 타일은 라벨 식별 정보 외에 아무
것도 담지 않으므로 이 위험을 설계 단계에서 원천 차단한다.

### Limitation
시각화만으로는 페이지 내용을 직관적으로 확인할 수 없다. 실제 내용 확인은
CSV 산출물이나 원본 PDF가 필요하다.

---

## 검토했으나 채택하지 않은 방식 (전체)

| 방식 | 미채택 이유 |
|---|---|
| 전 페이지 OCR | 39페이지 전부 텍스트 레이어 존재. 최소 68자, OCR 불필요 |
| TF-IDF + 전통 ML | INCOME 학습 샘플 1개, OTHER 0개. 과적합 확정. 데이터가 수천 페이지 쌓이면 가장 저렴한 정답 |
| LayoutLM / Donut 파인튜닝 | 성능 상한은 가장 높으나 GPU·라벨링·시간 부족 |
| 전 페이지 Vision LLM | 토큰 비용이 텍스트 대비 훨씬 크고 랜덤 회전 보정이 추가로 필요. 텍스트 레이어가 있는 데이터셋에서 낭비 |
| 전 페이지 LLM 위임 | 재현성 저하(실행마다 결과 변동), 비용, 속도. 룰이 37/39를 결정론적으로 확정 |
| 물리적 인접성만으로 원본 복원 | 인접 동일 유형 0/38쌍. 39개 단일 그룹만 산출 |
| 불확실 페이지 강제 결합 | 오귀속 위험. orphan으로 유지하는 것이 안전 |
| 벤더 주소를 시그니처로 사용 | 데이터셋 과적합. 주소 없이도 CREDIT 18페이지 전체 커버 |
| 무차별 숫자 마스킹 | p36의 유일한 분류 신호인 금액 구조를 파괴 |
| `rule_margin` 임계값 상향 | 데이터를 보고 임계값을 맞추는 과적합. p4/p11은 정답 페이지 |
| 중복 논리 페이지 임의 삭제 | 어느 쪽이 최신인지 판단 근거 없음. 둘 다 보존 + flag |
