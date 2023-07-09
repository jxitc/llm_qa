import os
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

class Retriever(object):
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
        
        
    def prepare_doc(self, init_query):
        """
        Only query once when use started the conversation
        """
        # TODO(xiao): here we start using PDF path for testing. 
        pdf_path = Retriever.get_pdf_path(init_query, self.allowlist)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split() # remove \n in parsed and splitted pdf
        self.db = self.__build_similarity_db(pages)

    def answer(self, query):
        if not query:
            return None  # empty user query
        similar_results = self.db.similarity_search(query)
        num_found_result = len(similar_results)
        print(f"found {num_found_result} similar docs for user query {query}")
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        # TODO(xiao): due to LLM word limitation, only use the first doc
        return chain.run(input_documents=similar_results[:1], question=query)
        
        
if __name__ == '__main__':
    qa = LlmQA([])  # TODO: empty allowlist
    init_query = u"这里面的数字化机会有哪些？"
    qa.prepare_doc(init_query)
    print(qa.answer(init_query))
