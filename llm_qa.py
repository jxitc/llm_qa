import os
import requests
from typing import Any, Union
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


class Retriever(object):
    @staticmethod
    def get_all(query):
        # TODO: set remote location in os env or config file
        response = requests.get('http://47.92.127.176:13240/xiyou/aiAnalysis/reptiles?query=' + query)
        return response.json()['data']  # TODO: error handling
        
    @staticmethod
    def get_url(query, allowlist):
        """
        base on user ``query`` search Search Baidu/Google for matching 
        similar result, filter by ``allowlist``, and returns as URL
        """
        return "http://www.moa.gov.cn/govpublic/FZJHS/202001/t20200120_6336316.htm"
        
    @staticmethod
    def get_pdf_path(query, allowlist):
        """
        base on user ``query`` search Search Baidu/Google for matching 
        similar result, filter by ``allowlist``, download it onto 
        local file and return its path
        """
        return "./test/test.pdf"

class UTF8WebBaseLoader(WebBaseLoader):
    def _scrape(self, url: str, parser: Union[str, None] = None) -> Any:
        # https://github.com/langchain-ai/langchain/issues/9925
        from bs4 import BeautifulSoup

        if parser is None:
            if url.endswith(".xml"):
                parser = "xml"
            else:
                parser = self.default_parser

        self._check_parser(parser)

        html_doc = self.session.get(url)
        html_doc.encoding = 'utf-8'
        return BeautifulSoup(html_doc.text, parser)


class LlmQA(object):
    def __init__(self, allowlist):
        self.allowlist = allowlist
        self.db = None

    @staticmethod
    def __build_similarity_db(docs):
        """
        build embedding db (in memory)
        TODO(xjiang): change to VectorDB in the future
        """
        embedder = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embedder)
        num_doc = len(docs)
        # TODO(xiao): use log
        print(f"done processing db, count: {num_doc}")
        return db

    @staticmethod
    def __split_all_urls(data_list):
      all_results = []
      for data in data_list:
        if data['uriType'] == 'webpage':
          loader = UTF8WebBaseLoader(data['uri'])
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size = 800,
              chunk_overlap  = 20,
              length_function = len,
          )
          result = loader.load_and_split(text_splitter)
          all_results += result
          print("added {} entries from {}".format(len(result), data['uri']))
        else:
          print("unsupported uriType: {}".format(data['uriType']))
      return all_results

    def prepare_doc(self, init_query):
        """
        Only query once when use started the conversation
        """
        all_url_data_list = Retriever.get_all(init_query)
        all_url_split_entries = self.__split_all_urls(all_url_data_list)
        self.db = self.__build_similarity_db(all_url_split_entries)

    def answer(self, query):
        if not query:
            return None  # empty user query

        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}
        
        Question: {question}
        Answer in Chinese:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        similar_results = self.db.similarity_search(query)
        num_found_result = len(similar_results)
        print(f"found {num_found_result} similar docs for user query {query}")
        debug_docs = []
        for r in similar_results:
            debug_docs.append(r.page_content.replace(' ', '').replace('\n', ''))
        chain = load_qa_chain(OpenAI(temperature=0, max_tokens=1024), chain_type="stuff", prompt=PROMPT)
        # TODO(xiao): due to LLM word limitation, only use the first doc
        return (chain.run(input_documents=similar_results[:2], question=query), debug_docs)

if __name__ == '__main__':
    qa = LlmQA([])  # TODO: empty allowlist
    init_query = u"如何解读数字农业农村发展规划"
    query = u" 请帮我分析医保政策"
    qa.prepare_doc(query)
    #query = u"这里面的数字化机会有哪些？"
    output, debug_docs = qa.answer(query)
    print("debug docs:")
    for doc in debug_docs:
        print("\n------")
        print(doc)
        print("------\n")
    print("output: " + output)
