import json
import configparser
import os
import time

from openai import OpenAI
import openai
from requests import Session
from typing import TypeVar, Generator
import io

from retry import retry
from tqdm import tqdm

from arxiv_scraper import get_papers_from_arxiv_rss_api
from filter_papers import filter_by_author, filter_by_gpt
from parse_json_to_md import render_md_string
from push_to_slack import push_to_slack
from arxiv_scraper import EnhancedJSONEncoder

T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> list[T]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_paper_batch(
    session: Session,
    ids: list[str],
    S2_API_KEY: str,
    fields: str = "paperId,title",
    **kwargs,
) -> list[dict]:
    params = {
        "fields": fields,
        **kwargs,
    }
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {
            "X-API-KEY": S2_API_KEY,
        }
    body = {
        "ids": ids,
    }

    with session.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        params=params,
        headers=headers,
        json=body,
    ) as response:
        response.raise_for_status()
        return response.json()


def translate_to_chinese_via_deepseek(text: str, client: OpenAI) -> str:
    """
    使用 DeepSeek API 将英文文本翻译成中文
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates English text to Chinese."},
                {"role": "user", "content": f"Translate the following text to Chinese:\n\n{text}"},
            ],
            stream=False,
            temperature=1.0,
            seed=0
        )
        translated_text = response.choices[0].message['content'].strip()
        return translated_text
    except Exception as e:
        print(f"翻译失败: {e}")
        return text


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")

    S2_API_KEY = os.environ.get("S2_KEY")
    OAI_KEY = os.environ.get("OAI_KEY")
    if OAI_KEY is None:
        raise ValueError(
            "OpenAI key is not set - please set OAI_KEY to your OpenAI key"
        )
    openai_client = OpenAI(api_key=OAI_KEY, base_url="https://api.deepseek.com")

    with io.open("configs/authors.txt", "r") as fopen:
        author_names, author_ids = parse_authors(fopen.readlines())
    author_id_set = set(author_ids)

    papers = list(get_papers_from_arxiv(config))

    all_authors = set()
    for paper in papers:
        all_authors.update(set(paper.authors))
    if config["OUTPUT"].getboolean("debug_messages"):
        print("Getting author info for " + str(len(all_authors)) + " authors")
    all_authors = get_authors(list(all_authors), S2_API_KEY)

    if config["OUTPUT"].getboolean("dump_debug_file"):
        with open(
            config["OUTPUT"]["output_path"] + "papers.debug.json", "w"
        ) as outfile:
            json.dump(papers, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(
            config["OUTPUT"]["output_path"] + "all_authors.debug.json", "w"
        ) as outfile:
            json.dump(all_authors, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(
            config["OUTPUT"]["output_path"] + "author_id_set.debug.json", "w"
        ) as outfile:
            json.dump(list(author_id_set), outfile, cls=EnhancedJSONEncoder, indent=4)

    selected_papers, all_papers, sort_dict = filter_by_author(
        all_authors, papers, author_id_set, config
    )
    filter_by_gpt(
        all_authors,
        papers,
        config,
        openai_client,
        all_papers,
        selected_papers,
        sort_dict,
    )

    # 增加翻译成中文的模块
    for paper_id, paper in selected_papers.items():
        print(f"Translating paper: {paper['title']}")
        paper['title_cn'] = translate_to_chinese_via_deepseek(paper['title'], openai_client)
        paper['abstract_cn'] = translate_to_chinese_via_deepseek(paper['abstract'], openai_client)
    
    # 排序论文
    keys = list(sort_dict.keys())
    values = list(sort_dict.values())
    sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
    selected_papers = {key: selected_papers[key] for key in sorted_keys}
    if config["OUTPUT"].getboolean("debug_messages"):
        print(sort_dict)
        print(selected_papers)

    # 推送到 Slack
    if len(papers) > 0:
        if config["OUTPUT"].getboolean("dump_json"):
            with open(config["OUTPUT"]["output_path"] + "output.json", "w") as outfile:
                json.dump(selected_papers, outfile, indent=4)
        if config["OUTPUT"].getboolean("dump_md"):
            with open(config["OUTPUT"]["output_path"] + "output.md", "w") as f:
                f.write(render_md_string(selected_papers))

            # 生成包含中文翻译的 Markdown 文件
            with open(config["OUTPUT"]["output_path"] + "output_translated.md", "w") as f:
                for paper_id, paper in selected_papers.items():
                    f.write(f"## {paper['title_cn']}\n\n")
                    f.write(f"{paper['abstract_cn']}\n\n")

        if config["OUTPUT"].getboolean("push_to_slack"):
            SLACK_KEY = os.environ.get("SLACK_KEY")
            if SLACK_KEY is None:
                print(
                    "Warning: push_to_slack is true, but SLACK_KEY is not set - not pushing to slack"
                )
            else:
                push_to_slack(selected_papers)
