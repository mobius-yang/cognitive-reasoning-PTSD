# data_clean.py
# 将doc数据进行清理，按照excel表格的提示内容整理成标准格式

import pandas as pd
from docx import Document
import re
from tqdm import tqdm
import os
from collections import defaultdict

def load_word_file(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def extract_main_context(full_text):
    category = None
    suds_before = None
    suds_after = None

    m = re.search(r"创伤事件(?:是)?：\s*([^\s，。；；\n\r]+)", full_text)
    if m:
        category = m.group(1).strip()

    m_pre = re.search(r"本次写作前的难受程度[:：]\s*(\d+)\s*分", full_text)
    if m_pre:
        suds_before = int(m_pre.group(1))

    m_post = re.search(r"本次写作完成后的难受程度[:：]\s*(\d+)\s*分", full_text)
    if m_post:
        suds_after = int(m_post.group(1))

    start_key = "本次写作前的难受程度"
    end_key = "本次写作完成后的难受程度"

    narrative = ""
    if start_key in full_text and end_key in full_text:
        start_idx = full_text.find(start_key)
        end_idx = full_text.find(end_key)
        narrative = full_text[start_idx:end_idx]
        narrative = narrative.split(start_key, 1)[-1].strip()

    return category, suds_before, suds_after, narrative


def extract_feedback(full_text):
    prompt = "写作反馈："
    if prompt in full_text:
        start_feedback = full_text.find(prompt) + len(prompt)
        feedback = full_text[start_feedback:].strip()
    else:
        feedback = None
    return feedback

def extract_comment(full_text):
    comments = None
    comments = re.findall(r"【(.*?)】", full_text, flags=re.S)
    return comments

def parse_file_name(filename):
    if '评语' in filename or '反馈' in filename:
        name, session, role = None, None, '评语'
    elif '誊录' in filename or '眷录' in filename or '第一阶段' in filename or '第二阶段' in filename:
        name, session, role = None, None, '誊录'
    else:
        raise ValueError(f"Your path now is {filename}, Filename must contain either '评语' or '誊录'")
    match = re.search(r"(.*?)-咨询(\d+)", filename)
    if match:
        name = match.group(1)
        session = int(match.group(2))
    else:
        match = re.search(r"(.*?)+咨询(\d+)", filename)
        if match:
            name = match.group(1)
            session = int(match.group(2))
    return name, session, role

def make_clean_data(data_path):
    session_data = defaultdict(lambda: {
        'category': None,
        'suds_before': None,
        'suds_after': None,
        'narrative': None,
        'feedback': None,
        'comments': None
    })
    skipped = []
    for fname in tqdm(os.listdir(data_path)):
        if not fname.endswith('.docx'):
            continue
        file_path = os.path.join(data_path, fname)
        full_text = load_word_file(file_path)
        name, session, role = parse_file_name(fname)
        if name is None or session is None:
            print("Skipped:", fname)
            skipped.append(file_path)
            continue

        key = (name, session)
        if role == '誊录':
            category, suds_before, suds_after, narrative = extract_main_context(full_text)
            session_data[key]['category'] = category
            session_data[key]['suds_before'] = suds_before
            session_data[key]['suds_after'] = suds_after
            session_data[key]['narrative'] = narrative
        elif role == '评语':
            feedback = extract_feedback(full_text)
            comments = extract_comment(full_text)
            session_data[key]['feedback'] = feedback
            session_data[key]['comments'] = comments

    records = []
    for (name, session), data in session_data.items():
        record = {
            'name': name,
            'session': session,
            'category': data['category'],
            'suds_before': data['suds_before'],
            'suds_after': data['suds_after'],
            'narrative': data['narrative'],
            'feedback': data['feedback'],
            'comments': data['comments']
        }
        records.append(record)
    
    return records, skipped

if __name__ == "__main__":
    data_paths = "D:\\vscode_install\\MyCode\\.vscode\\core\\积石山誊录稿"
    target_path = "D:\\vscode_install\\MyCode\\.vscode\\core\\cleaned_data.xlsx"
    all_records = []
    all_skipped = []

    for subdir in os.listdir(data_paths):
        subdir_path = os.path.join(data_paths, subdir)
        records, skipped = make_clean_data(subdir_path)
        all_records.extend(records)
        all_skipped.extend(skipped)

    df = pd.DataFrame(all_records)
    df.to_excel(target_path, index=False)
    print(f"Cleaned data saved to {target_path}")
    print(f"Skipped files: {all_skipped}")