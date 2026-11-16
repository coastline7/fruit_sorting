"""Русскоязычный интерфейс системы сортировки фруктов."""
import json
import logging
from pathlib import Path

import streamlit as st

from inference import FruitSystem, make_csv

ROOT = Path(__file__).resolve().parent
EXAMPLES = json.loads((ROOT / 'examples/manifest.json').read_text())
EXAMPLE_LABELS = {
    item['id']: f"{index + 1:02d} · {item['fruit']} · {item['quality']}"
    for index, item in enumerate(EXAMPLES)
}

st.set_page_config(page_title='Плод — сортировка фруктов', page_icon='🍏',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container{max-width:1390px;padding-top:2.4rem;padding-bottom:2rem}
h1{letter-spacing:-.055em;font-weight:650!important;line-height:1.13!important}
h2,h3{letter-spacing:-.025em}.brand{font-size:2.4rem;font-weight:750;letter-spacing:-.08em;color:#426044}
.eyebrow{font-size:.66rem;letter-spacing:.18em;color:#6f845f;font-weight:650;margin-bottom:.65rem}
.intro{color:#74806c;font-size:.97rem;line-height:1.8;margin-bottom:1.8rem}
[data-testid="stSidebar"]{border-right:1px solid #dfe7d7}
[data-testid="stForm"]{border-radius:16px;background:#fff;border-color:#dfe7d7}
[data-testid="stMetric"]{background:#eef3e7;border-radius:12px;padding:12px 18px}
[data-testid="stMetricValue"]{color:#49693e}
[data-testid="stBaseButton-primary"]{background:#496847;border-color:#496847;border-radius:9px}
[data-testid="stFileUploader"]{border-radius:12px}
.result-label{display:inline-block;padding:5px 12px;border-radius:20px;font-size:.8rem;font-weight:600}
.accept{color:#426c36;background:#eaf2e0}.reject{color:#a85945;background:#f9ece5}.review{color:#967327;background:#fbf2d7}
.empty{border:1px dashed #cdd9c0;border-radius:15px;min-height:310px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;background:#fbfcf8;padding:30px;color:#75846b}
.empty strong{font-size:1.15rem;color:#52664a;margin-bottom:10px}.empty p{font-size:.9rem;line-height:1.8;margin:0}
.footnote{font-size:.76rem;color:#7d8973;margin-top:2rem;border-top:1px solid #dfe7d7;padding-top:1rem}
@media(max-width:700px){.block-container{padding-top:1.4rem}h1{font-size:2.1rem!important}}
</style>''', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_system():
    # Общий экземпляр хранит модели, но не фотографии посетителей.
    # Замок внутри класса ограничивает одновременную обработку.
    return FruitSystem()


def example_entries(ids):
    allowed = {item['id'] for item in EXAMPLES}
    if any(item not in allowed for item in ids):
        raise ValueError('Неизвестный пример фотографии.')
    return [{'name': item, 'content': (ROOT / 'examples' / item).read_bytes()}
            for item in ids]


with st.sidebar:
    st.markdown('<div class="brand">плод.</div>', unsafe_allow_html=True)
    st.caption('Компьютерное зрение')
    st.divider()
    st.markdown('**Проверка фруктов**')
    st.caption('Яблоки · Бананы · Лаймы')
    st.info('Начните с готовой партии из 12 фотографий или загрузите свои снимки.')
    with st.expander('Как пользоваться'):
        st.markdown('''1. Добавьте фотографии или выберите примеры.
2. Выберите режим обработки.
3. При необходимости включите поиск участков на яблоках.
4. Нажмите кнопку проверки и скачайте таблицу результатов.''')
        st.caption('В контроле партии квота ручной проверки равна целой части от 10% количества фотографий. Для 1–9 фото квота равна нулю, для 10–19 — одному фото, для 20 — двум.')
    with st.expander('Как читать оценки'):
        st.write('Оценка дефекта — ответ модели. Она не означает процент повреждённой поверхности.')
        st.write('Обычная сортировка использует границу 50%. Контроль партии использует сохранённую исследовательскую границу 3%, три варианта яркости и ограниченную ручную проверку.')
        st.write('Рамки на яблоках показываются отдельно от решения о качестве. Номера 0 и 1 соответствуют исходной разметке. Их предметное значение не подтверждено.')
    with st.expander('Результаты исследования'):
        st.write('На 1343 контрольных изображениях нейронная сеть правильно определила внешнее качество в 99,78% случаев, вид фрукта — в 99,70%. Получены два пропуска дефектных изображений и одна ошибочная отбраковка.')
        st.write('Ручная проверка 134 фотографий устранила ошибки качества среди автоматически обработанных исходных снимков. Результат проверки человеком не моделировался. Условные затраты выросли с 0,0156 до 0,0200 на изображение.')
        st.write('Дополнительная польза индивидуальных границ и изменчивости ответа не установлена. Сохранённые границы одинаковы, вес изменчивости равен нулю.')
        st.write('Для обнаружения участков яблок качество при требовании совпадения рамок от 50% составило 77,69%.')
    with st.expander('Ограничения'):
        st.write('Поддерживаются только яблоки, бананы и лаймы. Неизвестный предмет тоже может получить один из этих ответов.')
        st.write('Оценка относится ко всей фотографии, включая группы плодов. Внутренние повреждения, химический состав и пригодность к употреблению не оцениваются.')
        st.write('Поиск участков проверен на яблоках с тёмным фоном. Для другой съёмки качество может отличаться. Для применения на линии требуется отдельная проверка.')
    st.divider()
    st.caption('Учебный исследовательский прототип')
    st.caption('Фотографии обрабатываются на сервере в памяти и не записываются приложением на диск.')

st.markdown('<div class="eyebrow">ПРОВЕРКА ВНЕШНЕГО КАЧЕСТВА</div>', unsafe_allow_html=True)
st.title('Сортировка фруктов по фотографии')
st.markdown('<div class="intro">Добавьте яблоки, бананы или лаймы. Система определит вид фрукта, оценит внешнее качество и предложит решение о сортировке.</div>', unsafe_allow_html=True)

left, right = st.columns([.9, 1.25], gap='large')

with left:
    with st.form('batch_form', clear_on_submit=False):
        st.subheader('01 · Фотографии и настройки')
        uploads = st.file_uploader('Загрузите фотографии', type=['jpg', 'jpeg', 'png', 'webp'],
                                   accept_multiple_files=True,
                                   help='До 20 фотографий, до 10 МБ каждая. Общий размер партии — до 50 МБ.')
        selected_examples = st.multiselect('Или выберите готовые примеры',
                                          options=list(EXAMPLE_LABELS),
                                          format_func=lambda value: EXAMPLE_LABELS[value],
                                          placeholder='Выберите фотографии из набора')
        mode_title = st.radio('Режим обработки', ['Обычная сортировка', 'Контроль партии'],
                              help='Контроль партии направляет до 10% фотографий на ручную проверку. Для партии меньше 10 фотографий эта квота равна нулю.')
        show_regions = st.checkbox('Показать участки на яблоках',
                                    help='Дополнительный поиск с рамками. Обработка может занять больше времени.')
        st.caption('Поиск участков и оценка качества выполняются разными моделями. Рамки не изменяют решение о сортировке.')
        submitted = st.form_submit_button('Проверить фотографии', type='primary', width='stretch')
        demo_submitted = st.form_submit_button('Проверить 12 примеров', width='stretch')
    st.caption('Готовая партия заменяет выбор файлов для одного запуска. Выбранный режим и поиск участков применяются к ней.')

if submitted or demo_submitted:
    st.session_state['result_filter'] = 'Все'
    st.session_state.pop('result', None)
    st.session_state.pop('processing_error', None)
    try:
        if demo_submitted:
            entries = example_entries([item['id'] for item in EXAMPLES])
        else:
            entries = example_entries(selected_examples)
            entries.extend({'name': item.name, 'content': item.getvalue()} for item in uploads)
        if not entries:
            raise ValueError('Добавьте фотографии или нажмите кнопку проверки 12 примеров.')
        with st.spinner('Обрабатываем фотографии. Первый запуск включает загрузку моделей в память.'):
            mode = 'batch' if mode_title == 'Контроль партии' else 'neural'
            result = get_system().analyze(entries, mode, show_regions)
        st.session_state['result'] = result
    except (ValueError, RuntimeError) as error:
        st.session_state['processing_error'] = str(error)
    except Exception:
        logging.exception('Ошибка обработки партии')
        st.session_state['processing_error'] = (
            'Обработка не завершена. Проверьте наличие папок models и config '
            'в репозитории. Подробности доступны владельцу приложения в журнале Community Cloud.'
        )

with right:
    st.subheader('02 · Результат проверки')
    if st.session_state.get('processing_error'):
        st.error(st.session_state['processing_error'])
    result = st.session_state.get('result')
    if not result:
        st.markdown('''<div class="empty"><strong>Здесь появятся решения о сортировке</strong><p>Загрузите свои фотографии или нажмите<br>кнопку проверки 12 примеров.<br><br>Принять · Отбраковать · Ручная проверка</p></div>''', unsafe_allow_html=True)
    else:
        columns = st.columns(3)
        for column, key, title in zip(columns, ['accepted', 'rejected', 'review'],
                                      ['Принять', 'Отбраковать', 'Ручная проверка']):
            column.metric(title, result['counts'][key])
        current_mode = 'Контроль партии' if result['mode'] == 'batch' else 'Обычная сортировка'
        st.caption(f"{len(result['items'])} фото · {result['seconds']:.2f} с · {current_mode}")
        if result['mode'] == 'batch' and len(result['items']) < 10:
            st.info('В этой партии меньше 10 фотографий. Квота ручной проверки равна нулю.')
        st.download_button('Скачать таблицу результатов', data=make_csv(result),
                           file_name='fruit_sorting_results.csv', mime='text/csv',
                           on_click='ignore', width='stretch')
        result_filter = st.radio('Показать результаты', ['Все', 'Принять', 'Отбраковать', 'Ручная проверка'],
                                 horizontal=True, key='result_filter')
        filtered = [item for item in result['items']
                    if result_filter == 'Все' or item['decision_label'] == result_filter]
        if not filtered:
            st.caption('Для выбранного решения фотографий нет.')
        for item in filtered:
            with st.container(border=True):
                category = ['accept', 'reject', 'review'][item['decision']]
                st.markdown(f'<span class="result-label {category}">{item["decision_label"]}</span>',
                            unsafe_allow_html=True)
                st.markdown(f"### {item['fruit']}")
                st.text(item['filename'])
                st.image(item['annotated'] or item['image'], width='stretch')
                st.progress(item['defect_score'], text=f"Оценка дефекта · {item['defect_score'] * 100:.1f}%")
                if item['fruit_score'] < .6:
                    st.warning('Вид определён неуверенно. Проверьте, что на фотографии поддерживаемый фрукт.')
                if item['regions_checked']:
                    st.caption(f"Найдено участков: {len(item['regions'])}. Номера рамок обозначают категории исходной разметки.")
                with st.expander('Сведения об ответе'):
                    st.write(f"Граница решения о дефекте: {item['threshold'] * 100:.1f}%")
                    st.write(f"Оценка уверенности в виде: {item['fruit_score'] * 100:.1f}%")
                    st.caption('Сведения об уверенности не являются независимой гарантией правильности ответа.')

st.divider()
st.subheader('Фотографии для самостоятельной проверки')
st.caption('12 настоящих фотографий из уже использованной контрольной части исследования. Подписи соответствуют категориям исходного набора.')
for start in range(0, len(EXAMPLES), 6):
    columns = st.columns(6)
    for column, item in zip(columns, EXAMPLES[start:start + 6]):
        with column:
            st.image(str(ROOT / 'examples' / item['id']), width='stretch')
            st.markdown(f"**{item['fruit']}**")
            st.caption(item['quality'])
st.markdown('[Источник фотографий](https://zenodo.org/records/17609211) · [Исходный набор FruitNet](https://data.mendeley.com/datasets/b6fftwbr2v/3)')
st.caption('Встроенные примеры предназначены для демонстрации. Для новой независимой проверки нужны фотографии, не использованные в исследовании.')
st.markdown('<div class="footnote">плод. · Учебная система сортировки фруктов · Выполнение моделей на центральном процессоре</div>', unsafe_allow_html=True)
