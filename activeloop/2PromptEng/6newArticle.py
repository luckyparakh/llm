from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain.output_parsers import PydanticOutputParser
from unittest.mock import Base
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import requests
from newspaper import Article
from langchain.prompts import PromptTemplate


def get_data(article_url: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

    s = requests.session()

    try:
        r = s.get(article_url, headers=headers, timeout=10)
        if r.status_code == 200:
            article = Article(article_url)
            article.download(input_html=r.text)
            article.parse()
            print(article.text)
            print(article.title)
            return article.title, article.text
    except:
        print("Error in fetching article")
        exit()


article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
# article_url="https://www.radisys.com/"

title, text = get_data(article_url)

template = """
You are very got at writing articles. You task is to write summary of a new article using below examples as reference: 

Use below example for reference:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
- Climate change is causing a rise in global temperatures.
- This leads to melting ice caps and rising sea levels.
- Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
- Artificial Intelligence (AI) has developed significantly over the past decade.
- AI is now used in multiple fields such as healthcare, finance, and transportation.
- The future of AI is promising but requires careful regulation.
========================================
Input:
Title: {title} 
Text: {text}
========================================
{format_instructions}
"""
# Can be added in place of formatted
# Please share output in below format and share summary in bullet list format:
# Title: {title}
# Summary:


class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted summary of the article")

    @field_validator("summary")
    def has_three_bullets(cls, v):
        if len(v) < 3:
            raise ValueError(
                "Summary should have alteast three bullet points.")
        return v


parser = PydanticOutputParser(pydantic_object=ArticleSummary)
p = PromptTemplate(
    input_variables=["title", "text"],
    template=template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

c = p | llm
print("-------------------------------")
r=c.invoke({"title": title, "text": text}).content.strip()
print(r)
print(parser.parse(r).summary)
