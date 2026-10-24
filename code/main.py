"""실행: streamlit run main.py"""
import io
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DATA_URL = 'https://raw.githubusercontent.com/srum-0916/data_personal_project/refs/heads/main/data/open_llm_leaderboard.csv'


@st.cache_data(ttl=3600)
def load_data():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    # 공백만 있는 칸도 결측값으로 취급합니다. 0과 False는 유지합니다.
    return pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig').replace(r'^\s*$', np.nan, regex=True)


def main():
    st.set_page_config(page_title='공개 언어 모델 데이터 기초통계', page_icon='📊', layout='wide')
    st.title('공개 언어 모델 데이터 기초통계')
    st.write('Open LLM Leaderboard 원본 데이터의 구성, 결측치, 수치 분포를 살펴봅니다.')
    st.markdown(f'[원본 CSV 보기]({DATA_URL})')
    if st.sidebar.button('최신 데이터 다시 불러오기'):
        load_data.clear()
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


if __name__ == '__main__':
    main()
