"""실행: streamlit run main.py"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DATA_URL = 'https://raw.githubusercontent.com/srum-0916/data_personal_project/refs/heads/main/data/open_llm_leaderboard.csv'


@st.cache_data(ttl=3600)
def load_data(filename='open_llm_leaderboard.csv'):
    # code 폴더와 저장소 최상위 어디에 두어도 data 폴더를 찾습니다.
    root = Path(__file__).resolve().parent
    for candidate in [root / 'data' / filename, root.parent / 'data' / filename]:
        if candidate.is_file():
            return pd.read_csv(candidate, encoding='utf-8-sig').replace(r'^\s*$', np.nan, regex=True)
    url = DATA_URL.rsplit('/', 1)[0] + '/' + filename
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    # 공백만 있는 칸도 결측값으로 취급합니다. 0과 False는 유지합니다.
    return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig').replace(r'^\s*$', np.nan, regex=True)


def render_original():
    st.title('공개 언어 모델 데이터 기초통계')
    st.write('Open LLM Leaderboard 원본 데이터의 구성, 결측치, 수치 분포를 살펴봅니다.')
    st.markdown(f'[원본 CSV 보기]({DATA_URL})')
    try:
        data = load_data()
    except Exception as exc:
        st.error(f'데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인하세요. {exc}')
        st.stop()
    numeric = data.select_dtypes(include='number').columns.tolist()
    total_missing = int(data.isna().sum().sum())
    cells = data.size
    cards = st.columns(4)
    cards[0].metric('행 수 (평가 기록)', f'{len(data):,}')
    cards[1].metric('열 수', f'{len(data.columns):,}')
    cards[2].metric('수치형 열 수', f'{len(numeric):,}')
    cards[3].metric('결측 셀 수', f'{total_missing:,}')
    st.caption(f'전체 셀 중 결측 비율: {total_missing/cells*100 if cells else 0:.2f}% · 완전히 동일한 중복 행: {int(data.duplicated().sum()):,}개')
    st.caption('각 행은 평가 기록입니다. 같은 모델이 여러 번 평가되었을 수 있습니다. 통계 계산에서 결측값은 제외하고, 점수 0은 포함합니다.')

    overview, stats, missing, distribution = st.tabs(['데이터 구성', '기초통계', '결측치', '분포 탐색'])
    with overview:
        st.subheader('열별 정보')
        info = pd.DataFrame({'열 이름': data.columns, '자료형': data.dtypes.astype(str).values,
                             '값이 있는 행 수': data.notna().sum().values,
                             '결측 행 수': data.isna().sum().values,
                             '서로 다른 값 수 (결측 제외)': data.nunique().values})
        st.dataframe(info, hide_index=True, width='stretch')
        st.subheader('원본 데이터 미리보기')
        count = st.selectbox('표시할 행 수', [10, 30, 100], index=0)
        st.dataframe(data.head(count), hide_index=True, width='stretch')
        st.caption('화면 표에는 HTML이 포함된 값도 원문 텍스트로 표시됩니다.')
    with stats:
        st.subheader('수치형 열의 기초통계')
        st.caption('표준편차는 표본 표준편차(ddof=1)입니다. 평가 점수·모델 크기 등은 단위가 서로 다르므로 열별로 해석하세요.')
        if numeric:
            summary = data[numeric].describe().T.rename(columns={
                'count': '유효 개수', 'mean': '평균', 'std': '표준편차', 'min': '최솟값',
                '25%': '1사분위수', '50%': '중앙값', '75%': '3사분위수', 'max': '최댓값'})
            st.dataframe(summary.round(3), width='stretch')
            st.download_button('기초통계표 CSV 다운로드', summary.to_csv(encoding='utf-8-sig').encode('utf-8-sig'),
                               file_name='basic_statistics.csv', mime='text/csv')
        else:
            st.info('수치형 열이 없습니다.')
        st.caption('참고: 모델 크기의 음수 등 의심스러운 값도 원본 통계에는 그대로 포함됩니다. 결측 처리나 이상값 제외는 별도 전처리 단계에서 결정해야 합니다.')
    with missing:
        st.subheader('열별 결측치')
        table = pd.DataFrame({'열 이름': data.columns, '결측 개수': data.isna().sum().values})
        table['결측 비율 (%)'] = table['결측 개수'] / len(data) * 100 if len(data) else 0
        table = table.sort_values('결측 개수', ascending=False)
        st.dataframe(table.round(2), hide_index=True, width='stretch')
        nonzero = table.loc[table['결측 개수'] > 0]
        if nonzero.empty:
            st.success('결측치가 없습니다.')
        else:
            st.plotly_chart(px.bar(nonzero, x='열 이름', y='결측 개수', hover_data=['결측 비율 (%)']), width='stretch')
        st.caption('빈 칸, 공백만 있는 칸, pandas가 기본 결측 표기로 인식하는 값(예: NaN, N/A)을 결측값으로 계산합니다.')
    with distribution:
        if numeric:
            selected = st.selectbox('살펴볼 수치형 열', numeric, index=numeric.index('IFEval') if 'IFEval' in numeric else 0)
            values = data.loc[np.isfinite(data[selected]), [selected]]
            if values.empty:
                st.info('그래프에 표시할 유효한 수치가 없습니다.')
            else:
                bins = st.slider('히스토그램 구간 수', 5, 80, 30)
                st.plotly_chart(px.histogram(values, x=selected, nbins=bins, labels={'count': '기록 수'}), width='stretch')
                st.plotly_chart(px.box(values, y=selected, points='outliers'), width='stretch')
                st.caption('상자 중앙선은 중앙값, 상자는 가운데 50%입니다. 수염 밖의 점은 사분위 범위의 1.5배 기준 이상치이며 삭제하지 않았습니다.')
        categorical = data.select_dtypes(exclude='number').columns.tolist()
        if categorical:
            selected_cat = st.selectbox('살펴볼 범주형 열', categorical, index=categorical.index('Type') if 'Type' in categorical else 0)
            frequencies = data[selected_cat].astype('string').fillna('(결측)').value_counts().rename_axis('값').reset_index(name='기록 수')
            st.caption('기록 수가 많은 상위 20개 값을 표시합니다. 원본의 True/False 값은 범주로 취급합니다.')
            st.dataframe(frequencies.head(20), hide_index=True, width='stretch')
            st.plotly_chart(px.bar(frequencies.head(20), x='값', y='기록 수'), width='stretch')


GROUP = '대화용 여부'
SIZE = '모델 크기(B)'
SCORES = ['지시 수행 점수', '추론 점수', '수학 점수']
ORDER = ['사전학습', '대화용']
COLORS = {'사전학습': '#64748B', '대화용': '#2563EB'}
SIZE_LABELS = ['3B 미만', '3~10B 미만', '10~30B 미만', '30~70B 미만', '70B 이상']


def prepare_topic(raw):
    required = ['모델명', GROUP, SIZE] + SCORES
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError('전처리 CSV에 필요한 열이 없습니다: ' + ', '.join(missing))
    data = raw.copy()
    data[GROUP] = data[GROUP].astype('string').str.strip()
    for col in [SIZE] + SCORES:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # 화면에서 제외한 행 수를 공개합니다. 원본 CSV 파일은 수정하지 않습니다.
    valid = data[GROUP].isin(ORDER) & np.isfinite(data[SIZE]) & (data[SIZE] > 0)
    for col in SCORES:
        valid &= np.isfinite(data[col]) & data[col].between(0, 100)
    return data.loc[valid].copy(), int((~valid).sum())


def add_size_groups(data):
    result = data.copy()
    result['크기 구간'] = pd.cut(result[SIZE], bins=[0, 3, 10, 30, 70, np.inf],
                                labels=SIZE_LABELS, right=False)
    return result


def render_topic():
    st.title('주제별 시각화 · 두 유형의 평가 점수 비교')
    st.write('대화용으로 학습된 AI와 사전학습 기본 AI는 평가 점수에서 어떤 차이를 보일까?')
    st.caption('전처리 데이터: data/llm_project_data_cleaned.csv · IFEval은 명시된 답변 조건의 수행 정도이며, 사실 정확성과는 다른 지표입니다.')
    try:
        raw = load_data('llm_project_data_cleaned.csv')
        data, excluded = prepare_topic(raw)
    except Exception as exc:
        st.error(f'전처리 데이터를 읽지 못했습니다. data 폴더의 파일명과 열 이름을 확인하세요. {exc}')
        return
    if data.empty:
        st.warning('시각화할 유효한 기록이 없습니다.')
        return
    counts = data[GROUP].value_counts().reindex(ORDER, fill_value=0)
    c1, c2, c3 = st.columns(3)
    c1.metric('분석 모델 수', f'{len(data):,}')
    c2.metric('사전학습', f"{counts['사전학습']:,}")
    c3.metric('대화용', f"{counts['대화용']:,}")
    st.caption(f'파일 {len(raw):,}행·{len(raw.columns)}열 중 유형·크기·점수 오류로 화면에서 제외한 행: {excluded:,}개')
    if data['모델명'].duplicated().any():
        st.warning('동일 모델명이 반복되어 있습니다. 현재 그래프는 행 단위이므로 전처리 파일을 확인하세요.')
    with st.expander('속성별 데이터 유형과 시각화 선정 이유'):
        st.dataframe(pd.DataFrame([
            ['모델명', '식별용 텍스트', '표', '개별 모델과 반복 기록 확인'],
            [GROUP, '범주형 · 명목형', '막대그래프 / 박스플롯의 집단', '유형 구성과 점수 차이'],
            [SIZE, '수치형', '크기 구간별 박스플롯', '비슷한 크기에서도 유형 차이가 유지되는지'],
            *[[score, '수치형', '유형별 박스플롯', '중앙값·산포·분포의 겹침'] for score in SCORES],
            ['모델 구조·수치 정밀도·대화 형식 사용', '범주형 · 명목형', '빈도표 (데이터 확인)', '평가 조건과 구성 확인'],
        ], columns=['속성', '데이터 유형', '표현 방법', '확인할 내용']), hide_index=True, width='stretch')

    st.subheader('1. 유형별 모델 수')
    frequencies = counts.rename_axis(GROUP).reset_index(name='모델 수')
    frequencies['비율 (%)'] = frequencies['모델 수'] / len(data) * 100
    fig = px.bar(frequencies, x=GROUP, y='모델 수', color=GROUP, text='모델 수',
                 color_discrete_map=COLORS, category_orders={GROUP: ORDER}, hover_data=['비율 (%)'])
    fig.update_layout(showlegend=False, height=340)
    fig.update_yaxes(rangemode='tozero')
    st.plotly_chart(fig, width='stretch')
    st.dataframe(frequencies.round(2), hide_index=True, width='stretch')
    st.info('두 집단의 수가 다르므로 단순 빈도 대신 점수 분포를 비교합니다. 이후 분류 평가에서는 다수 유형만 예측하는 기준선도 함께 확인해야 합니다.')

    st.subheader('2. 유형별 평가 점수 박스플롯')
    points = st.checkbox('모든 모델의 관측점 표시', value=False, key='topic_points')
    st.caption('중앙선: 중앙값 · 상자: 가운데 50% · 수염 밖 점: 사분위 범위 1.5배 기준 이상치. 점을 숨겨도 계산에는 모두 포함됩니다.')
    summary_rows = []
    for score, container in zip(SCORES, st.columns(3)):
        with container:
            fig = px.box(data, x=GROUP, y=score, color=GROUP, points='all' if points else 'outliers',
                         hover_name='모델명', hover_data=[SIZE], category_orders={GROUP: ORDER},
                         color_discrete_map=COLORS, title=score)
            fig.update_layout(showlegend=False, height=420, xaxis_title='모델 유형')
            fig.update_yaxes(range=[-3, 103])
            st.plotly_chart(fig, width='stretch')
        for group in ORDER:
            values = data.loc[data[GROUP] == group, score]
            summary_rows.append({'평가': score, '유형': group, '모델 수': len(values),
                                 '평균': values.mean(), '중앙값': values.median(),
                                 '1사분위수': values.quantile(.25), '3사분위수': values.quantile(.75),
                                 '표준편차': values.std()})
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary.round(2), hide_index=True, width='stretch')
    for score in SCORES:
        rows = summary[summary['평가'] == score].set_index('유형')
        if rows['모델 수'].min() > 0:
            gap = rows.loc['대화용', '중앙값'] - rows.loc['사전학습', '중앙값']
            st.write(f"• {score}: 중앙값은 사전학습 {rows.loc['사전학습', '중앙값']:.2f}점, "
                     f"대화용 {rows.loc['대화용', '중앙값']:.2f}점이며, 차이(대화용 − 사전학습)는 {gap:+.2f}점입니다.")
    st.caption('세 평가는 문항 구성과 난도가 다릅니다. 각 평가 안에서 유형을 비교하며, 평가 간 원점수 차이를 능력 차이의 크기로 직접 비교하지 않습니다.')

    st.subheader('3. 모델 크기 구간별 지시 수행 점수')
    grouped = add_size_groups(data)
    fig = px.box(grouped, x='크기 구간', y=SCORES[0], color=GROUP,
                 category_orders={'크기 구간': SIZE_LABELS, GROUP: ORDER},
                 color_discrete_map=COLORS, points='all' if points else 'outliers',
                 hover_name='모델명', hover_data=[SIZE])
    fig.update_layout(height=480, boxmode='group', legend_title='모델 유형')
    fig.update_yaxes(range=[-3, 103])
    st.plotly_chart(fig, width='stretch')
    group_summary = grouped.groupby(['크기 구간', GROUP], observed=False)[SCORES[0]].agg(
        모델수='count', 평균='mean', 중앙값='median').reset_index()
    st.dataframe(group_summary.round(2), hide_index=True, width='stretch')
    comparisons = []
    for label in SIZE_LABELS:
        rows = group_summary[group_summary['크기 구간'] == label].set_index(GROUP)
        n0, n1 = int(rows.loc['사전학습', '모델수']), int(rows.loc['대화용', '모델수'])
        gap = rows.loc['대화용', '중앙값'] - rows.loc['사전학습', '중앙값'] if min(n0, n1) else np.nan
        comparisons.append({'크기 구간': label, '사전학습 수': n0, '대화용 수': n1,
                            '중앙값 차이 (대화용−사전학습)': gap,
                            '해석 주의': '한 유형 없음' if min(n0, n1) == 0 else ('한 유형이 10개 미만' if min(n0, n1) < 10 else '두 유형 비교 가능')})
    st.dataframe(pd.DataFrame(comparisons).round(2), hide_index=True, width='stretch')
    st.caption('B는 10억 개 매개변수입니다. 구간은 [0,3), [3,10), [10,30), [30,70), [70,∞)이며 경계값은 오른쪽 구간에 속합니다. 10개는 주의를 돕는 표시 기준이며 통계적 유의성을 판정하는 기준이 아닙니다.')
    st.warning('같은 구간에도 크기가 다른 모델이 포함됩니다. 구간별 비교는 크기의 영향을 완전히 통제하지 않으며, 적은 모델로 구성된 집단은 결과가 불안정할 수 있습니다.')
    with st.expander('해석·한계와 분석 데이터 확인'):
        st.write('이 그래프는 공개 데이터 안의 관찰된 차이를 보여줍니다. 유형별 차이가 있더라도 대화용 학습의 인과 효과를 입증하지는 않습니다. 모델 구조·학습 데이터·평가 형식 등이 함께 영향을 줄 수 있습니다.')
        st.write('전처리 시 반복 평가 중 원본 순서의 첫 기록만 유지했으므로 평가 조건 정보가 일부 손실될 수 있습니다. 동일 계열 모델도 여러 개 포함되어 있습니다.')
        st.write('현재는 데이터 탐색 단계입니다. 기준선·분류 모델·테스트 성능은 아직 계산하지 않았습니다. 그래프만으로 분류 성능을 확정할 수 없습니다.')
        st.dataframe(data, hide_index=True, width='stretch')
    st.download_button('유형별 점수 요약표 다운로드', summary.to_csv(index=False).encode('utf-8-sig'),
                       file_name='type_score_summary.csv', mime='text/csv')


def main():
    st.set_page_config(page_title='공개 언어 모델 분석', page_icon='📊', layout='wide')
    with st.sidebar:
        st.title('공개 언어 모델 분석')
        page = st.radio('화면 선택', ['원본 데이터 · 기초통계', '전처리 데이터 · 주제별 시각화'], key='navigation')
        if st.button('최신 데이터 다시 불러오기'):
            load_data.clear()
        st.caption('원본의 구성 확인 → 전처리 데이터로 유형별 비교')
    if page == '원본 데이터 · 기초통계':
        render_original()
    else:
        render_topic()
    # 최종 발표 앱 확장: ① 질문 ② 데이터 보기 ③ 기준선 ④ 모델 결과 ⑤ 해석·한계.
    # 현재 두 화면은 ② 데이터 보기의 원본 탐색/전처리 후 시각화에 해당합니다.


if __name__ == '__main__':
    main()
