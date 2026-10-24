"""실행: streamlit run main.py | 기본 데이터: 사용자가 제공한 llm_project_data.csv."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 저장소의 data 폴더에 가공 CSV을 넣으세요.
DATA_FILENAME = 'llm_project_data.csv'
GROUP = '대화용 여부'
SCORE = '지시 수행 점수'
SIZE = '모델 크기(B)'
ORDER = ['사전학습', '대화용']
COLORS = {'사전학습': '#64748B', '대화용': '#2563EB'}


def load_default():
    app_dir = Path(__file__).resolve().parent
    candidates = [app_dir / 'data' / DATA_FILENAME,
                  app_dir.parent / 'data' / DATA_FILENAME]
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path, encoding='utf-8-sig')
    raise FileNotFoundError(
        f'저장소의 data 폴더에 {DATA_FILENAME} 파일을 넣어 주세요. '
        '파일 이름이 다르면 main.py의 DATA_FILENAME을 수정하세요.')


def prepare(raw):
    required = ['모델명', GROUP, SIZE, SCORE]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError('필수 열이 없습니다: ' + ', '.join(missing))
    data = raw.copy()
    data[GROUP] = data[GROUP].astype(str).str.strip().replace(
        {'0': '사전학습', '1': '대화용', '0.0': '사전학습', '1.0': '대화용'})
    for col in [SIZE, SCORE]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    valid = data[GROUP].isin(ORDER) & np.isfinite(data[SCORE]) & data[SCORE].between(0, 100)
    return data.loc[valid].copy(), int((~valid).sum())


def main():
    st.set_page_config(page_title='대화용 AI는 답변 조건을 더 잘 지킬까?', page_icon='📊', layout='wide')
    st.title('대화용 AI는 답변 조건을 더 잘 지킬까?')
    st.write('대화용으로 추가 학습된 모델과 사전학습 기본 모델의 지시 수행 점수(IFEval)를 비교합니다.')
    st.caption('IFEval은 답변의 형식·길이·키워드 등 명시된 조건을 지키는 정도를 평가합니다. 사실 정확성과는 다른 지표입니다.')
    with st.sidebar:
        st.header('데이터')
        uploaded = st.file_uploader('다른 가공 CSV로 변경하기 (선택)', type=['csv'])
        st.caption(f'기본 파일: data/{DATA_FILENAME}')
        show_points = st.checkbox('박스플롯에 모든 관측점 표시', value=False)
    try:
        raw = pd.read_csv(uploaded, encoding='utf-8-sig') if uploaded else load_default()
        data, removed = prepare(raw)
    except Exception as exc:
        st.error(f'데이터를 읽을 수 없습니다. CSV 파일과 열 이름을 확인하세요. {exc}')
        st.stop()
    if data.empty:
        st.error('분석 가능한 행이 없습니다.')
        st.stop()
    st.caption(f'입력 {len(raw):,}행 · 점수 비교 {len(data):,}행 · 유형 또는 점수 오류 제외 {removed:,}행')
    summary = data.groupby(GROUP)[SCORE].agg(['count', 'mean', 'median', 'std']).reindex(ORDER)
    cols = st.columns(3)
    for col, group in zip(cols[:2], ORDER):
        row = summary.loc[group]
        col.metric(f'{group} 평균', f"{row['mean']:.2f}점" if pd.notna(row['mean']) else '데이터 없음')
        col.caption(f"평가 기록 {int(row['count']) if pd.notna(row['count']) else 0:,}개")
    gap = summary.loc['대화용', 'mean'] - summary.loc['사전학습', 'mean']
    cols[2].metric('평균 차이 (대화용 − 사전학습)', f'{gap:+.2f}점' if pd.notna(gap) else '비교 불가')

    st.subheader('1. 두 유형의 점수 분포')
    box = px.box(data, x=GROUP, y=SCORE, color=GROUP, category_orders={GROUP: ORDER},
                 color_discrete_map=COLORS, points='all' if show_points else 'outliers',
                 hover_data=['모델명', SIZE])
    box.update_layout(showlegend=False, height=470, yaxis_title='IFEval 점수 (0~100)', xaxis_title='모델 유형')
    box.update_yaxes(range=[-3, 103])
    st.plotly_chart(box, width='stretch')
    st.caption('상자 중앙선은 중앙값, 상자는 가운데 50%의 점수입니다. 수염 밖의 점은 사분위 범위의 1.5배 기준 이상치이며 자동 삭제하지 않습니다.')
    st.dataframe(summary.rename(columns={'count': '기록 수', 'mean': '평균', 'median': '중앙값', 'std': '표준편차'}).round(2), width='stretch')

    st.subheader('2. 모델 크기와 점수의 관계 — 유형별 선형 회귀')
    reg = data.loc[np.isfinite(data[SIZE]) & (data[SIZE] > 0)].copy()
    st.caption(f'가로축: 모델 크기 (B = 10억 개 매개변수) · 세로축: IFEval 점수. 크기가 없거나 0 이하인 {len(data)-len(reg)}행은 이 그래프에서만 제외합니다.')
    if reg.empty:
        st.info('유효한 모델 크기가 없어 회귀선을 그릴 수 없습니다.')
    else:
        fig = px.scatter(reg, x=SIZE, y=SCORE, color=GROUP,
                         category_orders={GROUP: ORDER}, color_discrete_map=COLORS,
                         hover_name='모델명', opacity=0.45)
        fits = []
        for group in ORDER:
            part = reg.loc[reg[GROUP] == group]
            if len(part) < 2 or part[SIZE].nunique() < 2:
                st.info(f'{group}: 서로 다른 크기의 관측값이 부족해 회귀선을 생략합니다.')
                continue
            x, y = part[SIZE].to_numpy(), part[SCORE].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope*x + intercept
            total = np.sum((y-y.mean())**2)
            r2 = 1-np.sum((y-predicted)**2)/total if total > 0 else np.nan
            line_x = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(go.Scatter(x=line_x, y=slope*line_x+intercept, mode='lines',
                                     name=f'{group} 회귀선', line=dict(color=COLORS[group], width=3)))
            fits.append({'유형': group, '기록 수': len(part), '기울기 (점/B)': slope,
                         '절편': intercept, 'R² (현재 데이터 적합)': r2})
        fig.update_layout(height=500, xaxis_title='모델 크기 (B)', yaxis_title='IFEval 점수', legend_title='모델 유형 / 회귀선')
        st.plotly_chart(fig, width='stretch')
        if fits:
            st.dataframe(pd.DataFrame(fits).round(4), hide_index=True, width='stretch')
        st.caption('회귀식: 점수 = 기울기 × 모델 크기 + 절편. 선은 각 유형에서 관측된 크기 범위에만 표시합니다. R²는 같은 데이터에 맞춘 정도이며, 별도 테스트 데이터의 예측 성능이 아닙니다.')
    with st.expander('데이터와 해석할 때 주의할 점'):
        duplicate_count = int(data['모델명'].duplicated().sum())
        st.write(f'같은 모델명이 반복되는 추가 기록 {duplicate_count}행을 포함합니다. 각 행을 평가 기록으로 취급하며 임의로 중복을 삭제하지 않았습니다.')
        st.write('유형은 원본 데이터의 분류를 따릅니다. 평균 차이만으로 대화용 학습의 인과 효과를 증명할 수 없습니다. 모델 크기·구조·평가 형식과 같은 다른 조건도 영향을 줄 수 있습니다.')
        st.write('현재는 탐색용 그래프입니다. 학습/테스트 분할, 기준선 및 후보 모델 비교는 아직 수행하지 않았습니다.')
        st.dataframe(data, hide_index=True, width='stretch')
    st.markdown('데이터 출처: [Hugging Face Open LLM Leaderboard](https://huggingface.co/datasets/open-llm-leaderboard/contents) · 사용자가 제공한 가공본')


if __name__ == '__main__':
    main()
