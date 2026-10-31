"""실행: streamlit run code/main.py. 데이터는 ../data 또는 공개 GitHub CSV에서 읽습니다."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_URL = 'https://raw.githubusercontent.com/srum-0916/data_personal_project/main/data/'
SIZE, TYPE, TARGET = '모델 크기(B)', '모델 유형', '지시 수행 점수'
TYPES = ['사전학습', '대화용', '분야별 추가 학습', '모델 병합']
COLORS = dict(zip(TYPES, ['#64748B', '#2563EB', '#D97706', '#059669']))
LABELS = {'🟢 pretrained': '사전학습', '💬 chat models (RLHF, DPO, IFT, ...)': '대화용',
          '🔶 fine-tuned on domain-specific datasets': '분야별 추가 학습',
          '🤝 base merges and moerges': '모델 병합'}
RENAMES = {'fullname': '모델명', 'Type': TYPE, '#Params (B)': SIZE, 'IFEval': TARGET,
           'BBH': '추론 점수', 'MATH Lvl 5': '수학 점수', 'Architecture': '모델 구조',
           'Precision': '수치 정밀도', 'Chat Template': '대화 형식 사용'}
SEED = 42
CAP = 30.809


@st.cache_data(ttl=3600)
def load_data(filename):
    folder = Path(__file__).resolve().parent
    for p in [folder.parent / 'data' / filename, folder / 'data' / filename]:
        if p.is_file():
            return pd.read_csv(p, encoding='utf-8-sig').replace(r'^\s*$', np.nan, regex=True)
    r = requests.get(BASE_URL + filename, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding='utf-8-sig').replace(r'^\s*$', np.nan, regex=True)


def checked_data(raw):
    required = ['모델명', SIZE, TYPE, TARGET]
    absent = [c for c in required if c not in raw]
    if absent:
        raise ValueError('필수 열 없음: ' + ', '.join(absent))
    d = raw.copy()
    for col in [SIZE, TARGET]:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    valid = d[TYPE].isin(TYPES) & d['모델명'].notna() & np.isfinite(d[SIZE]) & d[SIZE].gt(0)
    valid &= np.isfinite(d[TARGET]) & d[TARGET].between(0, 100)
    if not valid.all():
        raise ValueError(f'유형·크기·점수·모델명에 유효하지 않은 {int((~valid).sum())}행이 있습니다. CSV 전처리를 확인하세요.')
    if d['모델명'].duplicated().any():
        raise ValueError('모델명이 중복됩니다. 동일 모델이 학습과 평가에 섞이지 않도록 전처리를 확인하세요.')
    if len(d) < 50 or d[TYPE].value_counts().min() < 10:
        raise ValueError('모델 평가를 위한 기록 또는 유형별 기록이 부족합니다.')
    return d.reset_index(drop=True)


def make_model(name, extras=()):
    if name == '평균 기준선':
        return DummyRegressor(strategy='mean')
    cat = [] if name == '선형 회귀 · 크기만' else [TYPE]
    transforms = [('수치', 'passthrough', [SIZE] + list(extras))]
    if cat:
        transforms.append(('유형', OneHotEncoder(categories=[TYPES], drop='first', handle_unknown='ignore', sparse_output=False), cat))
    prep = ColumnTransformer(transforms, remainder='drop')
    estimator = (RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=5,
                                       random_state=SEED, n_jobs=1)
                 if name == '랜덤 포레스트 · 크기+유형' else LinearRegression())
    return Pipeline([('변환', prep), ('예측', estimator)])


def metrics(y, pred):
    return mean_absolute_error(y, pred), r2_score(y, pred)


@st.cache_resource(show_spinner='회귀 모델을 학습하고 검증하는 중입니다…')
def fit_experiment(data):
    # 테스트는 후보 선택에 사용하지 않습니다. 유형별 비율을 보존합니다.
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=.2,
                                          random_state=SEED, stratify=data[TYPE])
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(train, train[TYPE]))
    names = ['평균 기준선', '선형 회귀 · 크기만', '선형 회귀 · 크기+유형', '랜덤 포레스트 · 크기+유형']
    rows, fitted, predictions = [], {}, {}
    for name in names:
        template = make_model(name)
        fold_mae = []
        for a, b in cv:
            model = clone(template).fit(train.iloc[a][[SIZE, TYPE]], train.iloc[a][TARGET])
            fold_mae.append(mean_absolute_error(train.iloc[b][TARGET], model.predict(train.iloc[b][[SIZE, TYPE]])))
        model = template.fit(train[[SIZE, TYPE]], train[TARGET])
        pred = model.predict(test[[SIZE, TYPE]])
        train_mae, _ = metrics(train[TARGET], model.predict(train[[SIZE, TYPE]]))
        mae, r2 = metrics(test[TARGET], pred)
        rows.append({'모델': name, '학습 MAE': train_mae, '교차검증 MAE': np.mean(fold_mae),
                     '교차검증 MAE 표준편차': np.std(fold_mae), '테스트 MAE': mae, '테스트 R²': r2})
        fitted[name], predictions[name] = model, pred
    scores = pd.DataFrame(rows).set_index('모델')
    selected = scores.drop(index='평균 기준선')['교차검증 MAE'].idxmin()
    result = test[['모델명', TYPE, SIZE, TARGET]].copy()
    result['예측 점수'] = predictions[selected]
    result['잔차 (실제−예측)'] = result[TARGET] - result['예측 점수']
    result['절대 오차'] = result['잔차 (실제−예측)'].abs()
    return {'scores': scores, 'models': fitted, 'predictions': predictions, 'selected': selected,
            'train_idx': train_idx, 'test_idx': test_idx, 'test': result,
            'training_mean': train[TARGET].mean()}


def chart(fig):
    fig.update_layout(font=dict(family='sans-serif'), margin=dict(t=45, b=35), legend_title_text='')
    st.plotly_chart(fig, width='stretch')


def question_page(data):
    st.title('① 질문')
    st.header('AI의 지시 수행 점수는 정말 모델 크기와 모델 유형만으로 예측할 수 있는가?')
    st.write('대형 모델이 늘 좋은 점수를 얻는지, 모델의 제작·추가 학습 유형을 알면 예측이 얼마나 나아지는지 살펴봅니다. 평균만 사용하는 예측과 회귀 모델을 비교하여 두 정보가 설명하는 부분과 남는 오차를 확인합니다.')
    st.info(f'분석 범위: 공개 언어 모델 네 유형 · 크기 0 초과 {CAP}B 이하 · 현재 {len(data):,}개. B는 10억 개 매개변수입니다.')
    st.subheader('무엇을 입력하고 무엇을 예측하나?')
    st.dataframe(pd.DataFrame([
        ['입력', SIZE, '수치형', '모델의 매개변수 수를 10억 단위로 표현'],
        ['입력', TYPE, '범주형', '사전학습·대화용·분야별 추가 학습·모델 병합'],
        ['예측 대상', TARGET, '수치형', 'IFEval 엄격 준수율 두 지표의 평균 × 100'],
    ], columns=['역할', '속성', '자료형', '뜻']), hide_index=True, width='stretch')
    st.write('모델 유형에는 학습 방법과 병합 방법이 함께 포함되어 있어, 이를 하나의 학습량이나 학습 순서로 해석하지 않습니다. 추론·수학 점수, 모델명, 구조, 정밀도, 대화 형식은 이번 예측 입력에 넣지 않습니다.')
    st.caption('IFEval은 개별 지시의 엄격 준수율과 질문 전체 조건의 엄격 준수율을 평균한 0~100점 지표입니다. 답변의 사실 정확성 전체를 뜻하지 않습니다.')
    st.markdown('[평가 방법](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about) · [IFEval 계산](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/911)')
    st.subheader('추가 실험')
    st.write('기본 입력에 추론(BBH)·수학(MATH Level 5) 점수를 각각 또는 함께 추가해 예측 성능을 비교합니다.')
    st.caption('추가 실험은 다른 평가 점수를 이미 알고 있을 때의 예측입니다. 기존 테스트 결과를 확인한 뒤 수행한 후속 탐색이므로 완전히 독립적인 최종 검증은 아닙니다.')


def basic_stats(d, key):
    c = st.columns(4)
    for container, label, value in zip(c, ['행 수', '열 수', '수치형 열', '결측 셀'],
                                       [len(d), len(d.columns), len(d.select_dtypes('number').columns), d.isna().sum().sum()]):
        container.metric(label, f'{int(value):,}')
    st.caption(f'완전히 동일한 중복 행: {d.duplicated().sum():,}개 · 빈 문자열·공백과 pandas 기본 결측 표기를 결측으로 인식합니다. 점수 0은 유지합니다.')
    t1, t2, t3, t4 = st.tabs(['미리보기·열 정보', '기초통계', '결측치', '분포'])
    with t1:
        st.dataframe(d.head(50), hide_index=True, width='stretch')
        st.dataframe(pd.DataFrame({'자료형': d.dtypes.astype(str), '유효 개수': d.notna().sum(),
                                   '결측 개수': d.isna().sum(), '고유값 수': d.nunique()}), width='stretch')
    with t2:
        nums = d.select_dtypes('number')
        if len(nums.columns):
            summary = nums.describe().T.rename(columns={'count': '유효 개수', 'mean': '평균', 'std': '표준편차', 'min': '최솟값', '25%': '1사분위수', '50%': '중앙값', '75%': '3사분위수', 'max': '최댓값'})
            st.dataframe(summary.round(3), width='stretch')
            st.download_button('기초통계표 다운로드', summary.to_csv().encode('utf-8-sig'), 'statistics.csv', key=key+'_download')
        st.caption('표준편차는 표본 표준편차입니다. 원본의 음수 모델 크기 등도 이 표에는 그대로 포함합니다.')
    with t3:
        missing = pd.DataFrame({'열': d.columns, '결측 수': d.isna().sum().values})
        missing['결측 비율 (%)'] = missing['결측 수'] / max(len(d), 1) * 100
        st.dataframe(missing.round(2), hide_index=True, width='stretch')
        if missing['결측 수'].sum():
            chart(px.bar(missing[missing['결측 수'] > 0], x='열', y='결측 수'))
    with t4:
        numeric = d.select_dtypes('number').columns.tolist()
        if numeric:
            target = TARGET if TARGET in numeric else 'IFEval'
            col = st.selectbox('수치형 열', numeric, index=numeric.index(target) if target in numeric else 0, key=key+'_number')
            values = d.loc[np.isfinite(d[col])]
            chart(px.histogram(values, x=col, nbins=30))
            chart(px.box(values, y=col, points='outliers'))
        categories = d.select_dtypes(exclude='number').columns.tolist()
        if categories:
            col = st.selectbox('범주형 열', categories, key=key+'_category')
            table = d[col].astype('string').fillna('(결측)').value_counts().head(20).rename_axis('값').reset_index(name='기록 수')
            chart(px.bar(table, x='값', y='기록 수'))


def data_page(data):
    st.title('② 데이터 보기')
    st.markdown('[출처: Hugging Face Open LLM Leaderboard contents](https://huggingface.co/datasets/open-llm-leaderboard/contents)')
    section = st.radio('살펴볼 내용', ['전처리 데이터', '원본 기초통계', '전처리 근거', '주제별 그래프'], horizontal=True)
    if section == '원본 기초통계':
        try:
            basic_stats(load_data('open_llm_leaderboard.csv'), 'original')
        except Exception as e:
            st.error(f'원본을 읽지 못했습니다: {e}')
    elif section == '전처리 데이터':
        basic_stats(data, 'cleaned')
        st.caption('모델명은 식별자, 크기·세 평가 점수는 수치형, 유형·구조·정밀도·대화 형식은 범주형입니다. 구조의 결측치는 현재 예측에 영향을 주지 않습니다.')
    elif section == '전처리 근거':
        st.dataframe(pd.DataFrame([
            ['원본', 0, 4576, '36열의 전체 평가 기록'], ['네 유형 선택', 74, 4502, '지속 사전학습·멀티모달·기타 제외'],
            ['유효성 점검', 3, 4499, '양수가 아닌 크기 제외; 세 점수의 유효 범위 점검'],
            ['유형 충돌 제외', 58, 4441, '원본 전체에서 같은 모델명에 여러 유형'],
            ['반복 평가 정리', 47, 4394, '원본 순서의 첫 기록 유지'],
            ['크기 범위 제한', 273, 4121, 'Q3+3IQR=30.809B 초과 제외'],
        ], columns=['단계', '제외 행', '남은 행', '근거']), hide_index=True, width='stretch')
        st.write('크기의 Q1=6.061B, Q3=12.248B, IQR=6.187B입니다. 점수가 높거나 낮다는 이유만으로 삭제하지 않았으며, IFEval Raw는 IFEval을 100으로 나눈 중복 정보여서 제외했습니다.')
        st.warning('크기 울타리 밖에도 정상 모델이 있습니다. 이것은 오류 제거의 증명이 아닌 범위 제한입니다. 현재 전처리 CSV의 기록이므로 데이터가 바뀌면 전처리 근거도 다시 확인해야 합니다.')
        if len(data) != 4121:
            st.warning('현재 CSV 행 수가 기록된 4,121행과 다릅니다. 전처리 기록을 갱신하세요.')
    else:
        counts = data[TYPE].value_counts().reindex(TYPES, fill_value=0).rename_axis(TYPE).reset_index(name='모델 수')
        chart(px.bar(counts, x=TYPE, y='모델 수', color=TYPE, text='모델 수', color_discrete_map=COLORS))
        chart(px.box(data, x=TYPE, y=TARGET, color=TYPE, category_orders={TYPE: TYPES}, color_discrete_map=COLORS, points='outliers'))
        st.caption('상자 중앙선은 중앙값, 상자는 가운데 50%입니다. 수염 밖 점도 분석에는 포함합니다.')
        chart(px.scatter(data, x=SIZE, y=TARGET, color=TYPE, hover_name='모델명', opacity=.45, color_discrete_map=COLORS))
        st.dataframe(data.groupby(TYPE)[TARGET].agg(['count', 'mean', 'median', 'std']).round(2), width='stretch')
        st.caption('유형별 분포와 크기–점수 관계를 살펴보는 탐색 그래프이며, 유형 차이의 인과 효과를 입증하지 않습니다.')


def split_table(data, result):
    a = data.iloc[result['train_idx']][TYPE].value_counts()
    b = data.iloc[result['test_idx']][TYPE].value_counts()
    return pd.DataFrame({'학습': a, '테스트': b}).reindex(TYPES).fillna(0).astype(int)


def baseline_page(data, result):
    st.title('③ 기준선')
    st.write('아무 특징도 보지 않고 모든 모델에 학습 데이터의 평균 점수를 답하는 방법입니다. 크기·유형 정보가 도움이 되는지 판단할 출발점입니다.')
    st.write(f"전체 {len(data):,}개 → 학습 {len(result['train_idx']):,}개 / 테스트 {len(result['test_idx']):,}개. 유형 비율을 보존한 8:2 분할이며 난수 시드는 {SEED}입니다.")
    st.dataframe(split_table(data, result), width='stretch')
    baseline = result['scores'].loc['평균 기준선']
    c = st.columns(3)
    c[0].metric('학습 평균 = 모든 예측값', f"{result['training_mean']:.2f}점")
    c[1].metric('테스트 MAE', f"{baseline['테스트 MAE']:.2f}점")
    c[2].metric('테스트 R²', f"{baseline['테스트 R²']:.3f}")
    fig = px.histogram(data.iloc[result['test_idx']], x=TARGET, nbins=30)
    fig.add_vline(x=result['training_mean'], line_dash='dash', annotation_text='학습 평균 기준선')
    chart(fig)
    st.write('MAE는 실제와 예측의 차이를 절댓값으로 계산한 평균이며 낮을수록 좋습니다. R²는 테스트 점수의 분산 대비 제곱오차를 평가하며 높을수록 좋고 음수도 가능합니다.')
    st.caption('테스트 R²의 0 기준은 테스트 평균 예측입니다. 학습 평균을 사용하는 우리 기준선의 테스트 R²는 정확히 0이 아닐 수 있습니다. 평가 데이터의 정답 평균을 예측값으로 사용하지 않습니다.')
    st.warning('행 단위 분할이라 같은 기본 모델에서 파생된 다른 모델이 양쪽에 있을 수 있습니다. ⑤에서 제작자 단위 분할을 보조 점검하며, 이것도 완전한 계열 분리는 아닙니다.')


def coefficient_table(result):
    model = result['models']['선형 회귀 · 크기+유형']
    reg = model.named_steps['예측']
    return pd.DataFrame({'항목': ['모델 크기 1B 증가'] + [t+' (사전학습 대비)' for t in TYPES[1:]],
                         '계수 (점)': reg.coef_})


def results_page(data, result):
    st.title('④ 모델 결과')
    experiment = st.radio('실험 선택', ['기본 실험', '추가 실험 · 추론·수학 점수'], horizontal=True)
    if experiment != '기본 실험':
        extra_results_page(data, result)
        return
    st.write('학습 데이터 내부에서 유형 비율을 보존한 3겹 교차검증을 수행했습니다. 평균 기준선을 제외한 후보 중 교차검증 MAE가 가장 낮은 모델을 선택하고, 동일한 별도 테스트 데이터에서 평가합니다.')
    st.success('교차검증으로 선택한 모델: ' + result['selected'])
    st.dataframe(result['scores'].round(4), width='stretch')
    st.caption('교차검증 표준편차는 3개 검증 폴드의 MAE 차이이며 신뢰구간이 아닙니다. 각 폴드 안에서 전처리 변환과 모델을 다시 학습합니다. 테스트 결과로 후보를 선택하지 않습니다.')
    with st.expander('모델 설정과 실제 분석 순서'):
        st.write('선형 회귀는 절편 포함 최소제곱법입니다. 유형은 사전학습을 기준으로 원핫 인코딩하며 크기는 B 단위 원값을 사용합니다. 랜덤 포레스트는 150그루, 최대 깊이 8, 잎 최소 표본 5, 시드 42로 고정하고 추가 튜닝하지 않습니다.')
        st.write('CSV 읽기 → 핵심 열 유효성 점검 → 8:2 분할 → 학습 데이터 안 3겹 검증 → 후보 선택 → 전체 학습분으로 재학습 → 테스트 평가 → 오차 해석 순서입니다. 예측값을 0~100으로 잘라 성능을 개선하지 않습니다.')
    bars = result['scores'][['학습 MAE', '교차검증 MAE', '테스트 MAE']].reset_index().melt(id_vars='모델', var_name='평가 구분', value_name='MAE (점)')
    chart(px.bar(bars, x='모델', y='MAE (점)', color='평가 구분', barmode='group'))
    name = st.selectbox('자세히 살펴볼 모델', list(result['scores'].index), index=list(result['scores'].index).index(result['selected']))
    points = data.iloc[result['test_idx']][['모델명', SIZE, TYPE, TARGET]].copy()
    points['예측 점수'] = result['predictions'][name]
    points['잔차 (실제−예측)'] = points[TARGET] - points['예측 점수']
    fig = px.scatter(points, x=TARGET, y='예측 점수', color=TYPE, hover_name='모델명', hover_data=[SIZE], color_discrete_map=COLORS, opacity=.6)
    lo = min(0, points['예측 점수'].min()); hi = max(100, points['예측 점수'].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], name='정확한 예측선', mode='lines', line=dict(color='black', dash='dash')))
    chart(fig)
    st.caption('대각선에 가까울수록 정확합니다. 위쪽은 과대예측, 아래쪽은 과소예측입니다.')
    fig = px.scatter(points, x='예측 점수', y='잔차 (실제−예측)', color=TYPE, hover_name='모델명', color_discrete_map=COLORS, opacity=.6)
    fig.add_hline(y=0, line_dash='dash'); chart(fig)
    st.caption('잔차가 양수면 실제보다 낮게 예측했습니다. 특정 점수대에 오차가 몰리는지 확인합니다.')
    st.subheader('선형 회귀를 식으로 해석하기')
    st.dataframe(coefficient_table(result).round(4), hide_index=True, width='stretch')
    intercept = result['models']['선형 회귀 · 크기+유형'].named_steps['예측'].intercept_
    st.write(f'예측 점수 = {intercept:.3f} + 크기 계수 × 모델 크기(B) + 해당 유형 계수')
    st.caption('사전학습의 유형 계수는 0입니다. 유형 간 기울기는 같다는 가정이며, 계수는 다른 입력을 고정한 관찰 관계이지 학습 방식의 인과 효과가 아닙니다. 크기 0의 절편은 범위 밖이므로 실질적 능력으로 해석하지 않습니다.')
    st.download_button('모델 비교표 CSV', result['scores'].to_csv().encode('utf-8-sig'), 'regression_scores.csv')
    st.download_button('선택 모델 테스트 예측 CSV', result['test'].to_csv(index=False).encode('utf-8-sig'), 'test_predictions.csv')


@st.cache_data(show_spinner='추가 분할·크기 범위 점검 중입니다…')
def robustness(data, selected, kind):
    if kind == '제작자 분할':
        d = data
        groups = d['모델명'].str.split('/').str[0]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(d, groups=groups))
        model = make_model(selected).fit(d.iloc[a][[SIZE, TYPE]], d.iloc[a][TARGET])
        pred = model.predict(d.iloc[b][[SIZE, TYPE]])
        baseline = np.repeat(d.iloc[a][TARGET].mean(), len(b))
        return pd.DataFrame([['제작자 분할 · 선택 모델', len(a), len(b), *metrics(d.iloc[b][TARGET], pred)],
                             ['제작자 분할 · 평균 기준선', len(a), len(b), *metrics(d.iloc[b][TARGET], baseline)]],
                            columns=['조건', '학습 수', '테스트 수', 'MAE', 'R²'])
    raw = load_data('open_llm_leaderboard.csv')
    d = raw[raw.Type.isin(LABELS)].copy()
    valid = d['#Params (B)'].gt(0) & np.isfinite(d['#Params (B)'])
    for col in ['IFEval', 'BBH', 'MATH Lvl 5']:
        valid &= np.isfinite(d[col]) & d[col].between(0, 100)
    conflicts = raw.groupby('fullname').Type.nunique().loc[lambda x:x > 1].index
    d = d[valid & ~d.fullname.isin(conflicts)].drop_duplicates('fullname').rename(columns=RENAMES)
    d[TYPE] = d[TYPE].map(LABELS)
    # 크기 제한의 영향만 비교: 큰 모델도 있는 하나의 고정 테스트 분할을 두 학습 조건에 공통 사용.
    a, b = train_test_split(np.arange(len(d)), test_size=.2, random_state=SEED, stratify=d[TYPE])
    training, test = d.iloc[a], d.iloc[b]
    rows = []
    for condition, train in [('전체 크기로 학습', training), ('30.809B 이하만 학습', training[training[SIZE] <= CAP])]:
        model = make_model(selected).fit(train[[SIZE, TYPE]], train[TARGET])
        pred = model.predict(test[[SIZE, TYPE]])
        for label, mask in [('공통 테스트 전체', np.ones(len(test), dtype=bool)), ('30.809B 이하', (test[SIZE] <= CAP).to_numpy()), ('30.809B 초과', (test[SIZE] > CAP).to_numpy())]:
            if mask.sum() > 1:
                rows.append([condition, label, len(train), int(mask.sum()), *metrics(test[TARGET].to_numpy()[mask], pred[mask])])
    return pd.DataFrame(rows, columns=['학습 조건', '테스트 범위', '학습 수', '테스트 수', 'MAE', 'R²'])


def save_note(index):
    st.session_state[f'note_draft_{index}'] = st.session_state[f'interpretation_{index}']


def interpretation_page(data, result):
    st.title('⑤ 해석·한계')
    st.caption('그래프와 수치를 확인한 뒤 직접 작성하세요. 입력한 글은 다운로드해 보관할 수 있습니다.')
    fields = ['기본 실험 해석', '추가 실험 해석', '최종 결론', '한계와 더 해 보고 싶은 것']
    notes = []
    for i, label in enumerate(fields):
        value = st.text_area(label, value=st.session_state.get(f'note_draft_{i}', ''), height=150,
                             key=f'interpretation_{i}', on_change=save_note, args=(i,))
        notes.append(label + '\n' + value)
    st.download_button('내 해석 다운로드', '\n\n'.join(notes).encode('utf-8-sig'),
                       file_name='my_interpretation.txt', mime='text/plain')
    st.caption('입력 내용은 서버 파일이나 GitHub에 저장되지 않습니다. 브라우저 세션이 끝나면 사라질 수 있으니 다운로드하세요.')


@st.cache_resource(show_spinner='추가 실험을 계산하는 중입니다…')
def fit_extra(data):
    required = ['추론 점수', '수학 점수']
    for col in required:
        if col not in data or not pd.to_numeric(data[col], errors='coerce').between(0, 100).all():
            raise ValueError(f'{col}의 유효한 0~100 점수가 필요합니다.')
    d = data.copy()
    d[required] = d[required].apply(pd.to_numeric)
    a, b = train_test_split(np.arange(len(d)), test_size=.2, random_state=SEED, stratify=d[TYPE])
    train, test = d.iloc[a], d.iloc[b]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(train, train[TYPE]))
    configurations = [('기본', []), ('추론 추가', ['추론 점수']), ('수학 추가', ['수학 점수']),
                      ('추론+수학 추가', required)]
    rows, predictions = [], {}
    for family, name in [('선형 회귀', '선형 회귀 · 크기+유형'),
                         ('랜덤 포레스트', '랜덤 포레스트 · 크기+유형')]:
        for label, extras in configurations:
            template = make_model(name, extras)
            cols = [SIZE, TYPE] + extras
            errors = []
            for x, y in folds:
                model = clone(template).fit(train.iloc[x][cols], train.iloc[x][TARGET])
                errors.append(mean_absolute_error(train.iloc[y][TARGET], model.predict(train.iloc[y][cols])))
            model = template.fit(train[cols], train[TARGET])
            pred = model.predict(test[cols])
            mae, r2 = metrics(test[TARGET], pred)
            key = family + ' · ' + label
            rows.append({'조합': key, '모델': family, '입력 조합': label,
                         '교차검증 MAE': np.mean(errors), '테스트 MAE': mae, '테스트 R²': r2})
            predictions[key] = pred
    table = pd.DataFrame(rows)
    selected = table.loc[table['교차검증 MAE'].idxmin(), '조합']
    return {'table': table, 'selected': selected, 'predictions': predictions,
            'test': test[['모델명', SIZE, TYPE, TARGET]].copy()}


def extra_results_page(data, basic):
    st.subheader('추가 실험: 추론·수학 점수가 예측을 보완할까?')
    st.caption('모든 조합에 크기와 유형이 포함됩니다. 기본 실험과 같은 8:2 분할·시드 42, 같은 3겹 교차검증과 모델 설정을 사용합니다.')
    st.caption('추론은 BBH 환산 점수, 수학은 MATH Level 5 점수입니다. 실제 활용 시 이 두 평가 점수를 미리 알아야 합니다.')
    try:
        result = fit_extra(data)
    except Exception as e:
        st.error(f'추가 실험을 수행하지 못했습니다: {e}')
        return
    table = result['table']
    st.dataframe(table.drop(columns='조합').round(3), hide_index=True, width='stretch')
    chart(px.bar(table, x='입력 조합', y='교차검증 MAE', color='모델', barmode='group',
                 title='입력 조합별 교차검증 MAE · 작을수록 좋음'))
    st.caption('후속 실험의 조합 선택은 학습 데이터 내부 교차검증 MAE를 사용합니다. 아래 테스트 수치는 기존 테스트를 재사용한 탐색 비교입니다.')
    selected = result['selected']
    st.write('교차검증 MAE가 가장 낮은 조합: ' + selected)
    option = st.selectbox('예측 그래프에서 볼 조합', table['조합'].tolist(),
                          index=table['조합'].tolist().index(selected), key='extra_model')
    row = table.set_index('조합').loc[option]
    base = table[(table['모델'] == row['모델']) & (table['입력 조합'] == '기본')].iloc[0]
    c = st.columns(3)
    c[0].metric('테스트 MAE', f"{row['테스트 MAE']:.2f}점")
    c[1].metric('테스트 R²', f"{row['테스트 R²']:.3f}")
    c[2].metric('같은 모델의 기본 입력 대비 MAE 감소', f"{base['테스트 MAE']-row['테스트 MAE']:+.2f}점")
    points = result['test'].copy()
    points['예측 점수'] = result['predictions'][option]
    points['절대 오차'] = (points[TARGET] - points['예측 점수']).abs()
    fig = px.scatter(points, x=TARGET, y='예측 점수', color=TYPE, hover_name='모델명',
                     hover_data=[SIZE], color_discrete_map=COLORS, opacity=.6)
    lo, hi = min(0, points['예측 점수'].min()), max(100, points['예측 점수'].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='정확한 예측선',
                             line=dict(color='black', dash='dash')))
    chart(fig)
    st.caption('대각선에 가까울수록 정확합니다. 위쪽은 과대예측, 아래쪽은 과소예측입니다.')
    with st.expander('유형별 오차·크게 틀린 사례'):
        bytype = points.groupby(TYPE).agg(모델수=('절대 오차', 'size'), MAE=('절대 오차', 'mean'))
        st.dataframe(bytype.round(3), width='stretch')
        st.dataframe(points.nlargest(10, '절대 오차').round(3), hide_index=True, width='stretch')
    st.download_button('추가 실험 비교표 다운로드', table.to_csv(index=False).encode('utf-8-sig'),
                       file_name='extra_experiment.csv', mime='text/csv')
    st.caption('해석은 ⑤ 해석·한계 화면에서 직접 작성하세요. 분석 범위는 네 유형·30.809B 이하이며 동일 계열 모델이 학습·테스트에 섞일 수 있습니다.')


def main():
    st.set_page_config(page_title='AI의 지시 수행 점수 예측', page_icon='📊', layout='wide')
    pages = ['① 질문', '② 데이터 보기', '③ 기준선', '④ 모델 결과', '⑤ 해석·한계']
    with st.sidebar:
        st.title('AI 지시 수행 점수 예측')
        page = st.radio('발표 흐름', pages)
        st.caption('모델 크기 + 모델 유형 → IFEval 점수')
        if st.button('데이터·학습 결과 새로고침'):
            load_data.clear(); fit_experiment.clear(); robustness.clear()
        st.caption('고정 시드 42 · 학습 80% / 테스트 20% · 학습 안 3겹 교차검증')
    try:
        data = checked_data(load_data('llm_project_data_cleaned.csv'))
    except Exception as e:
        st.error(f'전처리 CSV를 불러올 수 없습니다: {e}')
        st.stop()
    if data[SIZE].max() > CAP + 1e-9:
        st.error('현재 CSV가 기록된 30.809B 분석 범위를 초과합니다. 파일과 전처리 설명을 맞춰 주세요.')
        st.stop()
    if page == pages[0]:
        question_page(data)
    elif page == pages[1]:
        data_page(data)
    else:
        result = fit_experiment(data)
        if page == pages[2]:
            baseline_page(data, result)
        elif page == pages[3]:
            results_page(data, result)
        else:
            interpretation_page(data, result)


if __name__ == '__main__':
    main()
